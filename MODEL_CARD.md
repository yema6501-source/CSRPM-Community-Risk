# Model Card: CSRPM

## Model name

Community Social Risk Prediction Model (CSRPM)

## Task

Forecast a future **community-level observed event-burden class** from historical aggregate community indicators.

## Inputs

A chronological multivariate panel containing community identifiers, historical administrative activity, contextual socioeconomic indicators, and optional environmental/service signals.

## Core components

### 1. Spatial-social tensor factorization

A community × time × feature tensor is factorized with a masked Tucker objective. Community factors are regularized by the geographic graph Laplacian.

### 2. Adaptive dependency learning

Community embeddings produce a context-adaptive interaction matrix constrained by geographic adjacency.

### 3. Temporal encoding

A recurrent temporal module models the evolution of graph-enhanced community states.

### 4. Risk head

The network outputs logits for low, moderate, and high future aggregate burden.

### 5. Uncertainty and calibration

Temperature scaling and entropy-based diagnostics are provided. Monte-Carlo dropout utilities can estimate epistemic-style disagreement.

## Objective

The factorization implementation supports:

\[
\mathcal L =
\mathcal L_{\mathrm{pred}}
+\lambda_r \mathcal L_{\mathrm{recon}}
+\lambda_s \mathcal L_{\mathrm{spatial}}
+\lambda_2 \|\Theta\|_2^2.
\]

## Intended evaluation

- chronological train/validation/test split;
- repeated seeds;
- modern baselines;
- 95% confidence intervals;
- paired significance tests;
- calibration;
- robustness to missingness/noise;
- component ablation;
- subgroup diagnostics.

## Important interpretation boundary

The output is a model estimate of **future observed aggregate event burden**. It is not a probability that individual residents are dangerous, unstable, criminal, or politically problematic.

## Known limitations

1. Administrative event data reflect reporting and institutional processes.
2. Static socioeconomic data can become temporally stale.
3. Community boundaries are artificial aggregation units.
4. Spatial adjacency does not capture all social interactions.
5. Distribution shift can degrade calibration.
6. High predictive performance does not justify automated intervention.
7. Fairness metrics cannot resolve normative questions about legitimate use.

## Deployment status

Research prototype. Not validated for autonomous operational decision-making.
