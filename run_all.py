from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import classical_baselines, fit_predict_classical
from src.data_pipeline import ChicagoDataConfig, acquire_chicago_dataset
from src.evaluation import classification_metrics
from src.fairness_audit import subgroup_audit, vulnerability_quartiles
from src.feature_engineering import (
    LeakageSafePreprocessor,
    add_temporal_features,
    build_future_target,
    chronological_boundaries,
    discretize_risk,
    fit_risk_thresholds,
    infer_feature_columns,
    split_mask,
)
from src.statistical_tests import holm_adjust, paired_wilcoxon, summarize_runs
from src.synthetic_data import SyntheticConfig, generate_synthetic_panel
from src.utils import ensure_dir, load_yaml, save_json, seed_everything
from src.visualization import save_calibration, save_confusion, save_metric_summary


def load_panel(mode: str, cfg: dict, runtime_data: Path) -> pd.DataFrame:
    if mode == "real":
        path = runtime_data / "chicago_panel.csv"
        if not path.exists():
            dc = cfg["datasets"]["chicago"]
            out = acquire_chicago_dataset(
                runtime_data,
                ChicagoDataConfig(
                    start_date=str(dc["start_date"]),
                    end_date=str(dc["end_date"]),
                    frequency=str(dc["frequency"]),
                    page_size=int(dc["page_size"]),
                    timeout=int(dc["timeout"]),
                ),
                app_token=os.getenv("CHICAGO_APP_TOKEN"),
            )
            return out["panel"]
        return pd.read_csv(path, parse_dates=["date"])
    sc = cfg["datasets"]["synthetic"]
    return generate_synthetic_panel(SyntheticConfig(**sc))


def prepare(panel: pd.DataFrame, horizon: int):
    count_features = [
        c for c in ["sanitation_count", "abandoned_vehicle_count", "observed_event_count"]
        if c in panel.columns
    ]
    df = add_temporal_features(panel, count_features=count_features)
    df = build_future_target(df, horizon=horizon)
    split = chronological_boundaries(df["date"])
    masks = split_mask(df["date"], split)

    thresholds = fit_risk_thresholds(df.loc[masks["train"], "future_event_count"].to_numpy())
    df["risk_class"] = discretize_risk(df["future_event_count"].to_numpy(), thresholds)

    feature_cols = infer_feature_columns(df)
    X_raw = df[feature_cols].to_numpy(float)
    prep = LeakageSafePreprocessor().fit(X_raw[masks["train"]])
    X = prep.transform(X_raw)
    y = df["risk_class"].to_numpy(int)
    return df, X, y, feature_cols, masks, thresholds, split


def run_classical_experiments(df, X, y, masks, seeds, results_dir: Path):
    rows = []
    prediction_cache = {}
    for seed in seeds:
        seed_everything(seed)
        for name, model in classical_baselines(seed).items():
            pred, prob = fit_predict_classical(model, X[masks["train"]], y[masks["train"]], X[masks["test"]])
            metrics = classification_metrics(y[masks["test"]], prob)
            rows.append({"model": name, "seed": seed, **metrics})
            prediction_cache[(name, seed)] = (pred, prob)

    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "baseline_seed_metrics.csv", index=False)

    summaries = []
    for model, group in table.groupby("model"):
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr_macro", "pr_auc_macro", "ece", "brier"]:
            stats = summarize_runs(group[metric].dropna().to_numpy())
            summaries.append({"model": model, "metric": metric, **stats})
    pd.DataFrame(summaries).to_csv(results_dir / "baseline_summary_95ci.csv", index=False)

    # Paired significance relative to the strongest mean Macro-F1 classical baseline.
    mean_f1 = table.groupby("model")["macro_f1"].mean()
    reference = mean_f1.idxmax()
    pvals = {}
    ref = table[table.model == reference].sort_values("seed")["macro_f1"].to_numpy()
    for model in mean_f1.index:
        if model == reference:
            continue
        other = table[table.model == model].sort_values("seed")["macro_f1"].to_numpy()
        pvals[model] = paired_wilcoxon(ref, other)["p_value"]
    save_json({"reference": reference, "raw_p": pvals, "holm_adjusted_p": holm_adjust(pvals)}, results_dir / "paired_significance.json")
    return table, prediction_cache


def run_fairness(df, y, masks, pred, results_dir):
    test = df.loc[masks["test"]].reset_index(drop=True)
    hardship = test.get("hardship_index", pd.Series(np.zeros(len(test))))
    quartile = vulnerability_quartiles(pd.to_numeric(hardship, errors="coerce").fillna(pd.to_numeric(hardship, errors="coerce").median()).to_numpy())
    audit = subgroup_audit(y[masks["test"]], pred, quartile)
    audit.to_csv(results_dir / "fairness_by_hardship_quartile.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="CSRPM end-to-end reproducibility runner.")
    parser.add_argument("--mode", choices=["real", "synthetic"], default="real")
    parser.add_argument("--runtime-data", default="runtime_data")
    parser.add_argument("--results", default="runtime_results")
    parser.add_argument("--quick", action="store_true", help="Use three seeds for a fast integrity run.")
    args = parser.parse_args()

    runtime_data = ensure_dir(args.runtime_data)
    results_dir = ensure_dir(args.results)
    cfg = {
        "datasets": load_yaml("configs/datasets.yaml"),
        "experiments": load_yaml("configs/experiments.yaml"),
    }
    seeds = cfg["experiments"]["seeds"]
    if args.quick:
        seeds = seeds[:3]

    panel = load_panel(args.mode, cfg, runtime_data)
    panel["date"] = pd.to_datetime(panel["date"])
    horizon = int(cfg["experiments"]["forecast_horizon_steps"])
    df, X, y, feature_cols, masks, thresholds, split = prepare(panel, horizon)

    protocol = {
        "mode": args.mode,
        "feature_columns": feature_cols,
        "risk_thresholds_fit_on_training_only": thresholds,
        "train_end": split.train_end,
        "validation_end": split.valid_end,
        "n_train": int(masks["train"].sum()),
        "n_valid": int(masks["valid"].sum()),
        "n_test": int(masks["test"].sum()),
        "seeds": seeds,
        "important": "The script does not contain hard-coded manuscript performance values.",
    }
    save_json(protocol, results_dir / "protocol.json")

    table, cache = run_classical_experiments(df, X, y, masks, seeds, results_dir)
    best = table.groupby("model")["macro_f1"].mean().idxmax()
    pred, prob = cache[(best, seeds[0])]
    run_fairness(df, y, masks, pred, results_dir)

    save_confusion(y[masks["test"]], pred, results_dir / "confusion_matrix.png")
    save_calibration(y[masks["test"]], prob, results_dir / "calibration.png")
    save_metric_summary(table, "macro_f1", results_dir / "baseline_macro_f1.png")

    print("CSRPM reproducibility integrity pipeline completed.")
    print(f"Mode: {args.mode}")
    print(f"Chronological split: train <= {split.train_end.date()}, validation <= {split.valid_end.date()}, test afterwards")
    print(f"Training-only risk thresholds: {thresholds}")
    print(f"Results directory: {results_dir.resolve()}")
    print("Note: deep CSRPM/GNN/Transformer training components are provided in src/ and configured separately;")
    print("the integrity runner intentionally avoids fabricating expensive-model results when they have not been executed.")


if __name__ == "__main__":
    main()
