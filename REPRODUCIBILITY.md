# Reproducibility Guide

## Reviewer-to-code traceability

| Concern | Executable evidence |
|---|---|
| Synthetic-only evaluation | `download_data.py`, `src/data_pipeline.py` |
| Real community validation | Chicago panel constructed from public administrative data |
| Missing tensor definition | `src/tensor_factorization.py` |
| Tensor optimization | `SupervisedTucker.objective()` |
| Spatial structure | `build_spatial_adjacency()` + graph Laplacian |
| Adaptive dependencies | `AdaptiveSpatialLayer` |
| Temporal dynamics | `CSRPMNet.temporal` |
| Training configuration | `configs/csrpm.yaml` |
| Baseline configuration | `configs/baselines.yaml` |
| Chronological split | `src/feature_engineering.py` |
| Leakage control | `LeakageSafePreprocessor` |
| Training-only target thresholds | `fit_risk_thresholds()` |
| Modern model families | baseline config for GNN/Transformer models |
| Ablation specification | `src/ablation.py` |
| Multiple seeds | `configs/experiments.yaml` |
| 95% CI | `bootstrap_mean_ci()` |
| Significance tests | `paired_wilcoxon()` |
| Multiple-comparison control | `holm_adjust()` |
| Uncertainty | `src/uncertainty.py` |
| Calibration | ECE, Brier, NLL, temperature scaling |
| Fairness | `src/fairness_audit.py` |
| Ethics/misuse | `ETHICS_AND_RESPONSIBLE_USE.md` |
| Numerical consistency | generated CSV/PNG outputs; no manuscript values hard-coded |

## Minimal software check

```bash
python -m compileall src
```

## Real-data acquisition

```bash
python download_data.py
```

Expected local files:

```text
runtime_data/chicago_panel.csv
runtime_data/chicago_adjacency.csv
runtime_data/chicago_socioeconomic.csv
runtime_data/chicago_community_areas.geojson
```

## Integrity experiment

```bash
python run_all.py --mode real --quick
```

This runs a leakage-safe repeated classical-baseline evaluation and produces machine-generated metrics, confidence intervals, statistical comparisons, calibration plots, and a basic fairness audit.

## Full experimental protocol

The full paper revision should execute the neural CSRPM, GNN, Transformer, and all ablations with the same chronological partitions and seed list. Unexecuted model values must not be manually inserted into result tables.

## Numerical-reporting rule

Every number included in a final manuscript table should have a traceable origin:
- raw seed-level result;
- aggregation script;
- uncertainty interval;
- test statistic where a significance claim is made.

If repository output and manuscript text disagree, repository output should be treated as the value requiring investigation.

## Environment capture

For an archival run, save:

```bash
python --version
pip freeze > environment_freeze.txt
```

The base dependency set is pinned in `requirements.txt`.

## Reproduction record

A rigorous experimental run should preserve:
- repository commit identifier;
- configuration files;
- seeds;
- data acquisition dates;
- data checksums;
- split boundaries;
- hardware information;
- run logs;
- seed-level metrics;
- final aggregate tables.

## No silent fallback

If real data cannot be acquired, the real-data pipeline raises an error. It does not automatically substitute simulated data.

## Controlled synthetic run

```bash
python run_all.py --mode synthetic --quick
```

This is suitable for software verification and controlled stress tests only.
