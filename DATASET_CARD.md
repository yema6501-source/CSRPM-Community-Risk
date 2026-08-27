# Dataset Card

## Purpose

The repository separates two data regimes:

1. **Real-world Chicago community-area panel** — principal empirical validation.
2. **Controlled synthetic panel** — sensitivity, ablation, missingness, and shock-response testing.

Synthetic observations must never be described as external validation.

## Real-world sources

### Chicago Crimes - 2001 to Present

Dataset identifier: `ijzp-q8t2`

Use: future aggregate event burden and historical event features.

Fields retained by default:
- date;
- community area;
- primary incident type;
- arrest indicator;
- domestic indicator.

Fields such as names or direct personal identifiers are not required.

### 311 Sanitation Code Complaints

Dataset identifier: `me59-5fac`

Use: aggregate community service-pressure signal.

### 311 Abandoned Vehicles

Dataset identifier: `3c9v-pnva`

Use: aggregate infrastructure/service signal where available.

### Selected Socioeconomic Indicators

Dataset identifier: `kn9c-c2s2`

Use: contextual community covariates including poverty, unemployment, education, dependency, per-capita income, and hardship index.

Important limitation: these indicators summarize an older census period and should be treated as slow-moving contextual covariates rather than current measurements.

### Community-area boundaries

Use: construction of a geographic adjacency prior for spatial regularization.

## Unit of analysis

The model operates on **community × time-window** observations.

It does not require an individual-level prediction target.

## Temporal construction

Raw incidents and service requests are aggregated using the frequency specified in `configs/datasets.yaml`.

Historical lags and rolling summaries are shifted so that information from a target period cannot leak into the predictor.

## Target

The target is future aggregate incident burden:

\[
Y_{c,t+h} = \text{reported event count for community } c \text{ at future horizon } t+h.
\]

For categorical evaluation, low/moderate/high boundaries are estimated from the training target distribution only.

## Missing data

Missing covariates are handled by a training-fitted median imputer with missingness indicators. Future work can compare multiple imputation, temporal interpolation, GRU-D-style models, or probabilistic imputation.

## Known biases

Open administrative datasets are not neutral measurements of latent social risk. Reported incidents may reflect:
- reporting behavior;
- enforcement patterns;
- access to 311 services;
- institutional practices;
- geographic differences in data coverage;
- changes in data systems over time.

Accordingly, the target is described as **observed administrative event burden**, not objective community dangerousness.

## Digital-divide warning

Communities with lower digital-service access can produce fewer online/311 records even when underlying need is substantial. Model output therefore cannot be interpreted as a direct measure of social worth, safety, cohesion, or legitimacy.

## Intended use

Research into aggregate forecasting, uncertainty estimation, robustness, and methodological comparison.

## Out-of-scope use

- individual profiling;
- predictive policing of persons;
- immigration enforcement;
- political surveillance;
- automated benefit denial;
- punitive allocation based solely on model output;
- labeling neighborhoods as inherently dangerous.
