---
icon: material/target
---

# Probes (Supervised)

Probes are **supervised** concept methods: they require labeled data (concept annotations)
to learn a mapping from activations to concept scores. This contrasts with the unsupervised
[Concept Spaces](./concept_spaces/base.md) (ICA, NMF, SAEs, …) which discover concepts
from unlabeled activations alone.

The probe workflow uses the same splitter activation extraction as unsupervised
methods, but the `fit` step requires both activations **and** binary concept labels.

## Usage Guide

### Classification Model (SplitterForClassification)

The simplest setup for classification models. `SplitterForClassification`
automatically detects the classification head and returns the single
representation consumed by that head.

```python
from interpreto import SplitterForClassification
from interpreto.concepts import ProbeExplainer
from interpreto.concepts.probes import LinearRegressionProbe

# 1. Wrap your classification model
model = SplitterForClassification("textattack/bert-base-uncased-imdb")

# 2. Extract classification representations - shape (n, d)
activations, predictions = model.get_activations(texts)

# 3. Instantiate probe and explainer
probe = LinearRegressionProbe()
explainer = ProbeExplainer(model, concept_model=probe)

# 4. Fit on activations + binary concept labels - labels shape (n, c)
explainer.fit(activations, labels)

# 5. Score new inputs
concept_scores = explainer.activations_to_concepts(new_activations)
```

### Text Model (TextTokensSplitter)

For causal and encoder text models, choose whether to keep token-level
activations or aggregate them into a fixed-size representation.

#### Strategy A: Aggregate to one vector per sample

Use `token_pooling` to pool the retained tokens into one activation vector per input.
This is appropriate when concepts are global properties of the input (e.g., topic, style).

```python
from interpreto import TextTokensSplitter
from interpreto.concepts import ProbeExplainer
from interpreto.concepts.probes import CosineCentroidProbe

model = TextTokensSplitter(
    "gpt2",
    split_point="transformer.h.6",
    task="text-generation",
    device_map="cuda",
)

# Aggregate retained tokens into one vector per sample - shape (n, d).
activations, _ = model.get_activations(texts, token_pooling="mean")

probe = CosineCentroidProbe()
explainer = ProbeExplainer(model, concept_model=probe)
explainer.fit(activations, labels)
```

#### Strategy B: Per-token activations (flattened)

The default output contains one activation per retained token, flattened across
samples. This is appropriate when concepts are local properties
(e.g., named-entity type, part-of-speech) and labels are provided per-token.

```python
# One vector per retained token - shape (n_tokens, d).
activations, _ = model.get_activations(texts)

# Labels must have one row per retained token: shape (n_tokens, c).
probe = LinearRegressionProbe()
explainer = ProbeExplainer(model, concept_model=probe)
explainer.fit(activations, token_labels)
```

!!! tip "Choosing a representation"
    - **Classification or pooled text:** one score per input, suitable for document-level concepts.
    - **Unpooled text:** one score per retained token, suitable for word-level concepts.

### Using Normalizations

Normalizations standardize or decorrelate activations before the probe sees them.
They are fitted jointly during `probe.fit()` and applied automatically at encode time.

```python
from interpreto.concepts.probes import (
    LinearRegressionProbe,
    CosineCentroidProbe,
    Standardization,
    Whitening,
)

# Zero-mean, unit-variance per feature
probe = LinearRegressionProbe(normalization=Standardization())

# SVD-based whitening (full rank)
probe = CosineCentroidProbe(normalization=Whitening())

# Low-rank whitening — projects to top-128 principal components
probe = CosineCentroidProbe(normalization=Whitening(rank=128))
```

### Using Bias Calibrators

Bias calibrators set the additive bias of a probe *after* fitting the weights/centroids.
They control the decision threshold: a sample is considered positive for concept *j*
when `score_j + bias_j > 0`.

```python
from interpreto.concepts.probes import (
    LinearRegressionProbe,
    DotProductCentroidProbe,
    prevalence_bias,
    fpr_bias,
    midpoint_bias,
    bce_bias,
    lda_shared_var_bias,
)

# Set threshold based on class prevalence (logit of prior)
probe = DotProductCentroidProbe(bias_calibrator=prevalence_bias)

# Control false-positive rate at 1%
probe = LinearRegressionProbe(bias_calibrator=fpr_bias)

# Midpoint between positive and negative score means
probe = LinearRegressionProbe(bias_calibrator=midpoint_bias)

# Optimize BCE loss on the intercept only
probe = LinearRegressionProbe(bias_calibrator=bce_bias)

# Bayes-optimal threshold assuming Gaussian class-conditionals
probe = DotProductCentroidProbe(bias_calibrator=lda_shared_var_bias)
```

### Combining Normalization + Bias Calibration

Both options compose naturally:

```python
from interpreto.concepts.probes import (
    CosineCentroidProbe,
    Standardization,
    fpr_bias,
)

probe = CosineCentroidProbe(
    normalization=Standardization(),
    bias_calibrator=fpr_bias,
)
# The pipeline at fit/encode time is:
#   1. Standardize activations (fitted during probe.fit)
#   2. Compute cosine similarity to centroids
#   3. Add calibrated bias (threshold at 1% FPR)
```

---

## ProbeExplainer

::: interpreto.concepts.ProbeExplainer
    handler: python
    options:
      show_root_heading: true
      show_source: true
      inherited_members: true
      members:
        - fit
        - activations_to_concepts
        - interpret
        - get_inputs_to_concepts_model

## Probe Models

All probes follow the `Probe` interface and can be passed as `concept_model`
to `ProbeExplainer`.

### Linear Probes

::: interpreto.concepts.probes.LinearRegressionProbe
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.probes.LogisticRegressionProbe
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.probes.LinearSVMProbe
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.probes.MeansDiffProbe
    handler: python
    options:
      show_root_heading: true

### Centroid Probes

::: interpreto.concepts.probes.CosineCentroidProbe
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.probes.DotProductCentroidProbe
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.probes.SqL2CentroidProbe
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.probes.SVDDCentroidProbe
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.probes.DiagonalMahalanobisCentroidProbe
    handler: python
    options:
      show_root_heading: true

### Normalizations

Normalizations can be composed with any probe to standardize or whiten the input activations
before probing.

::: interpreto.concepts.probes.Standardization
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.probes.Whitening
    handler: python
    options:
      show_root_heading: true

### Bias Calibrators

Post-hoc functions to set the bias of a fitted probe based on different criteria.

::: interpreto.concepts.probes.bce_bias
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.probes.fpr_bias
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.probes.prevalence_bias
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.probes.lda_shared_var_bias
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.probes.midpoint_bias
    handler: python
    options:
      show_root_heading: true

## Using other models with `sklearn`

::: interpreto.concepts.probes.sklearn.SklearnProbeExplainer
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: interpreto.concepts.probes.sklearn.SklearnProbe
    handler: python
    options:
      show_root_heading: true
      show_source: true
