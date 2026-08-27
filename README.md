# CSRPM Community Risk Reproducibility

Reproducibility package for **A Big Data–Informed Framework for Community Social Risk Prediction and Management**.

This repository is designed to make the computational claims auditable. It separates controlled simulation from real-world community validation, defines the spatial-social tensor mechanism explicitly, enforces chronological evaluation, discloses experiment settings, and provides statistical, uncertainty, robustness, and fairness utilities.

## Scientific scope

CSRPM is a **community-level forecasting system**. It is not an individual-risk classifier and is not intended to infer dangerousness, identity, political orientation, or future behavior of a person.

The corrected computational hypothesis is:

> Community risk can be forecast from temporally ordered, privacy-preserving aggregate indicators when higher-order community-time-feature structure, spatial relationships, temporal dynamics, and prediction uncertainty are modeled jointly.

The repository therefore treats the original small synthetic experiment as a controlled sensitivity test only. Real-world evaluation is built around public Chicago community-area data.

## Repository layout

```text
.
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── synthetic_data.py
│   ├── feature_engineering.py
│   ├── tensor_factorization.py
│   ├── csrpm_model.py
│   ├── baselines.py
│   ├── uncertainty.py
│   ├── evaluation.py
│   ├── statistical_tests.py
│   ├── ablation.py
│   ├── fairness_audit.py
│   ├── visualization.py
│   └── utils.py
├── configs/
│   ├── csrpm.yaml
│   ├── datasets.yaml
│   ├── baselines.yaml
│   └── experiments.yaml
├── run_all.py
├── download_data.py
├── requirements.txt
├── DATASET_CARD.md
├── MODEL_CARD.md
├── ETHICS_AND_RESPONSIBLE_USE.md
├── REPRODUCIBILITY.md
├── CODE_AVAILABILITY.md
├── CITATION.cff
└── .gitignore
```

Only two source folders are maintained. Runtime datasets and results are generated locally and excluded from version control.

## Real-world data design

The principal real-world experiment uses public Chicago community-area information.

### Target source

**Chicago Crimes - 2001 to Present**  
City of Chicago Data Portal dataset: `ijzp-q8t2`

The repository aggregates reported incidents by community area and time window. The prediction target is **future aggregate incident burden**, not an author-created latent score.

### Auxiliary signals

- 311 sanitation-code requests: `me59-5fac`
- 311 abandoned-vehicle requests: `3c9v-pnva`
- Chicago selected socioeconomic indicators: `kn9c-c2s2`
- Chicago community-area geographic boundaries

The data are aggregated to community level before modeling. Direct personal identifiers are not required.

Public data portal:
`https://data.cityofchicago.org/`

## Leakage-safe forecasting task

For every community \(c\) and time \(t\), model input contains data observed no later than \(t\). The target is event burden at a future horizon \(t+h\).

\[
\hat{y}_{c,t+h}
=
f(X_{\leq t}, A, Z_c)
\]

where:

- \(X_{\leq t}\) is the historical multivariate community panel,
- \(A\) is the community adjacency matrix,
- \(Z_c\) contains static community-level contextual indicators,
- \(h\) is the forecast horizon.

Risk classes are derived from **training-period target quantiles only**. Validation/test targets never influence thresholds, imputation statistics, or scaling parameters.

## Spatial-social tensor definition

The principal tensor is

\[
\mathcal{X} \in \mathbb{R}^{N_c \times N_t \times N_f},
\]

with community, time, and feature modes.

A Tucker representation is learned:

\[
\hat{\mathcal{X}}
=
\mathcal{G}
\times_1 U_c
\times_2 U_t
\times_3 U_f.
\]

The implementation is located in `src/tensor_factorization.py`.

The joint objective is:

\[
\mathcal{L}
=
\mathcal{L}_{pred}
+
\lambda_r
\left\|
M \odot
(\mathcal{X}-\hat{\mathcal{X}})
\right\|_F^2
+
\lambda_s
\operatorname{Tr}(U_c^\top L_s U_c)
+
\lambda_2 \|\Theta\|_2^2.
\]

Here \(M\) masks unobserved entries and \(L_s\) is the graph Laplacian derived from real community boundaries.

This makes the factorization testable: removing it is an ablation, and its reconstruction/spatial terms are independently observable.

## Adaptive spatial-temporal predictor

`src/csrpm_model.py` defines an adaptive graph-constrained interaction layer and a recurrent temporal encoder.

The adaptive dependency layer learns community similarity only along graph-supported neighborhoods plus self connections. This prevents an unconstrained dense affinity matrix from being interpreted as geographic evidence.

The temporal module receives graph-enhanced community states and produces the final categorical logits.

## Uncertainty

Raw class confidence alone is insufficient for high-impact community decision support. `src/uncertainty.py` provides:

- predictive entropy;
- normalized entropy;
- temperature scaling;
- expected calibration error;
- multiclass Brier score;
- negative log likelihood;
- Monte-Carlo dropout predictive entropy;
- expected entropy;
- mutual information.

Interpretation is deliberately conservative:

| Predicted risk | Uncertainty | Interpretation |
|---|---|---|
| High | Low | stronger model evidence; still requires human/context review |
| High | High | obtain additional evidence before escalation |
| Low | High | insufficient evidence to dismiss potential concern |
| Low | Low | stable low-risk model estimate, not a guarantee |

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Download real data

```bash
python download_data.py
```

An optional City of Chicago Socrata application token may reduce public API throttling:

```bash
export CHICAGO_APP_TOKEN="..."
python download_data.py
```

No token is committed to this repository.

Generated files are placed in `runtime_data/`.

## One-command integrity experiment

```bash
python run_all.py --mode real
```

For a short software-integrity run:

```bash
python run_all.py --mode real --quick
```

Controlled synthetic experiment:

```bash
python run_all.py --mode synthetic --quick
```

The synthetic experiment must not be reported as real-world external validation.

## Expected runtime outputs

`runtime_results/` contains:

- `protocol.json`
- `baseline_seed_metrics.csv`
- `baseline_summary_95ci.csv`
- `paired_significance.json`
- `fairness_by_hardship_quartile.csv`
- `confusion_matrix.png`
- `calibration.png`
- `baseline_macro_f1.png`

No result number is hard-coded into the source code. Manuscript tables should be generated from executed result files.

## Baselines

The repository defines classical baselines directly and provides configuration space for modern neural/graph models:

- Logistic Regression
- Random Forest
- Gradient Boosting
- MLP
- LSTM
- GRU
- GCN
- GraphSAGE
- GAT
- temporal Transformer

The purpose of broad baselines is not to maximize the number of comparisons. It is to test whether CSRPM improves over:
1. simple linear decision boundaries,
2. nonlinear tabular models,
3. temporal models,
4. graph models,
5. attention-based temporal models.

## Ablation protocol

`src/ablation.py` defines:

1. full CSRPM;
2. no tensor factorization;
3. no spatial regularization;
4. static rather than adaptive dependency;
5. no temporal encoder;
6. no uncertainty calibration.

A claim that a component is necessary should be supported by repeated-run ablation results rather than architectural intuition alone.

## Statistical protocol

Default seeds:

`13, 29, 41, 67, 83, 101, 127, 149, 173, 199`

Reported model comparisons should include:

- mean;
- standard deviation;
- 95% bootstrap confidence interval;
- paired Wilcoxon test across matched seeds;
- Holm correction for multiple baseline comparisons;
- effect-size reporting where appropriate.

`src/statistical_tests.py` implements the core procedures.

## Fairness and subgroup diagnostics

Because community prediction can amplify structural inequality, evaluation includes performance stratification across contextual vulnerability groups. The package supports:

- subgroup Macro-F1;
- high-risk true-positive rate;
- high-risk false-positive rate;
- high-risk false-negative rate;
- worst-group performance;
- calibration/uncertainty comparisons by subgroup.

These diagnostics do not establish that a deployment is fair. They expose disparities that require substantive review.

## Reproducibility rules

1. Training, validation, and test periods are chronological.
2. Preprocessing fits on training rows only.
3. Risk-class thresholds are learned from training targets only.
4. Random seeds are disclosed.
5. All reported values should come from stored result files.
6. Failure to acquire real data is treated as an error; it is not silently replaced by synthetic data.
7. Synthetic results are labeled as synthetic.
8. Community identifiers are geographic aggregate units, not individuals.
9. No raw personal text or personally identifying field is required.
10. The repository does not assert unexecuted performance.

## Computational limitations

The full spatial-temporal model is more expensive than classical tabular baselines. Tensor reconstruction cost scales with community, time, feature, and rank dimensions. The implementation therefore separates the integrity pipeline from expensive full-model training so that a repository reader can inspect and reproduce each computational component without fabricated benchmark output.

## Manuscript-result consistency

The current package is intended to become the computational source of truth. After experiments are executed, manuscript accuracy, F1, feature influence, uncertainty, ablation, and statistical tables should be replaced by values generated from the repository. Manually entered numbers should not be retained when they disagree with executable output.

## Responsible use

Read `ETHICS_AND_RESPONSIBLE_USE.md` before adapting the system. Predictions are decision-support signals only. They must not be used for individual targeting, automatic punitive action, political surveillance, or stigmatization of neighborhoods.

## Reproduction checklist

See `REPRODUCIBILITY.md` for a reviewer-oriented mapping from methodological claims to files and executable evidence.
