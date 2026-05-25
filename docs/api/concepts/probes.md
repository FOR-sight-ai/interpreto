---
icon: material/target
---

# Probes (Supervised)

Probes are **supervised** concept methods: they require labeled data (concept annotations)
to learn a mapping from activations to concept scores. This contrasts with the unsupervised
[Concept Spaces](./concept_spaces/base.md) (ICA, NMF, SAEs, …) which discover concepts
from unlabeled activations alone.

The probe workflow uses the same `ModelWithSplitPoints` activation extraction as unsupervised
methods, but the `fit` step requires both activations **and** binary concept labels.

## Usage Guide

### Classification Model (SplitSequenceClassification)

The simplest setup for classification models. `SplitSequenceClassification` automatically
detects the classification head and uses CLS-token activations.

```python
from interpreto import SplitSequenceClassification
from interpreto.concepts import ProbeExplainer
from interpreto.concepts.probes import LinearRegressionProbe

# 1. Wrap your classification model
model = SplitSequenceClassification("textattack/bert-base-uncased-imdb")

# 2. Extract CLS-token activations — shape (n, d)
activations = model.get_activations(texts)

# 3. Instantiate probe and explainer
probe = LinearRegressionProbe()
explainer = ProbeExplainer(model, concept_model=probe)

# 4. Fit on activations + binary concept labels — labels shape (n, c)
explainer.fit(activations, labels)

# 5. Score new inputs
concept_scores = explainer.encode_activations(new_activations)
```

### Generation Model (ModelWithSplitPoints)

For generation models, you must choose how to aggregate the sequence of token-level
activations into a fixed-size representation. Two common strategies:

#### Strategy A: Aggregate to one vector per sample

Use `activation_granularity=SAMPLE` to pool all tokens into one activation vector.
This is appropriate when concepts are global properties of the input (e.g., topic, style).

```python
from interpreto import ModelWithSplitPoints
from interpreto.concepts import ProbeExplainer
from interpreto.concepts.probes import CosineCentroidProbe

model = ModelWithSplitPoints(
    "gpt2",
    split_points="transformer.h.6",
    device_map="cuda",
)

# Aggregate all tokens into one vector per sample — shape (n, d)
activations = model.get_activations(
    texts,
    activation_granularity=model.activation_granularities.SAMPLE,
    aggregation_strategy=model.aggregation_strategies.MEAN, # MAX and LAST are also often compared in the literature
)

probe = CosineCentroidProbe()
explainer = ProbeExplainer(model, concept_model=probe)
explainer.fit(activations["transformer.h.6"], labels)
```

#### Strategy B: Per-token activations (flattened)

Use `activation_granularity=TOKEN` to get one activation per token (special tokens
removed, then flattened). This is appropriate when concepts are local properties
(e.g., named-entity type, part-of-speech) and labels are provided per-token.

```python
# One vector per token, flattened across all samples — shape (n*l, d)
activations = model.get_activations(
    texts,
    activation_granularity=model.activation_granularities.TOKEN,
)

# labels must also be flattened to match: shape (n*l, c)
probe = LinearRegressionProbe()
explainer = ProbeExplainer(model, concept_model=probe)
explainer.fit(activations["transformer.h.6"], token_labels)
```

!!! tip "Choosing a granularity"
    - **SAMPLE / CLS_TOKEN**: one score per input — good for document-level concepts.
    - **TOKEN / WORD / SENTENCE**: one score per unit — good for local/fine-grained concepts.

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
        - encode_activations
        - interpret
        - inputs_to_concepts

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
