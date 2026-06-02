---
icon: material/transit-connection-variant
---

# Concept Attributions (Input-to-Concept)

Concept attributions answer the question: **which parts of the input are responsible for activating a given concept?**

This is achieved by combining the concept framework with the attribution framework:
a fitted concept explainer exposes an `inputs_to_concepts` property that returns a model
mapping raw inputs to concept activations. This model can then be passed to any
perturbation-based attribution method (Lime, KernelShap, Occlusion, Sobol).

## Overview

The input-to-concept attribution pipeline has three steps:

1. **Fit a concept explainer** on model activations (as in the standard concept pipeline).
2. **Get the bridge model** via `concept_explainer.inputs_to_concepts`.
3. **Run an attribution method** using the bridge model and the model's tokenizer.

The result is a per-token attribution for each selected concept.

## Quick Example

```python
import torch
from interpreto import Occlusion, SplitterForClassification
from interpreto.concepts import SemiNMFConcepts

# 1. Setup the split model
splitter = SplitterForClassification(
    "nateraw/bert-base-uncased-emotion",
    batch_size=32,
    device_map="cuda",
)

# 2. Fit a concept explainer
activations, predictions = splitter.get_activations(train_texts, tqdm_bar=True)
concept_explainer = SemiNMFConcepts(splitter, nb_concepts=20, device="cuda")
concept_explainer.fit(activations)

# 3. Compute input-to-concept attributions
explainer = Occlusion(
    concept_explainer.inputs_to_concepts,
    splitter.tokenizer,
    batch_size=256,
)

# Explain all concepts for a single input
results = explainer.explain("The stock market rallied on strong earnings.")

# Or explain specific concepts only
results = explainer.explain("Some text.", targets=torch.arange(5))

# See tutorials for visualization examples
```

## How It Works

The `inputs_to_concepts` property returns a `ModelForInputsToConcepts` object that:

1. Passes inputs through the model backbone via `SplitterForClassification.inputs_to_activations`
   to obtain CLS-token representations.
2. Encodes those latent activations into concept space using the fitted concept model's encoder.
3. Returns concept activations as pseudo-logits, which the attribution method treats as outputs.

The attribution method then perturbs input tokens and measures the change in concept activations,
producing token-level importance scores for each concept.

## Supported Attribution Methods

Only **perturbation-based** methods are supported:

| Method | Description |
|--------|-------------|
| `Occlusion` | Masks tokens one at a time |
| `Lime` | Fits a local linear model on perturbed inputs |
| `KernelShap` | SHAP values via weighted linear regression |
| `Sobol` | Sobol sensitivity indices |

Gradient-based methods (Saliency, IntegratedGradients, etc.) are **not compatible** because
the pipeline involves `nnsight` tracing which breaks the gradient tape.

## Targets

The `targets` parameter in `explainer.explain(inputs, targets=...)` specifies **which concepts**
to explain:

- `targets=None`: Explain all concepts (default).
- `targets=torch.arange(5)`: Explain concepts 0 through 4.
- `targets=torch.tensor([2, 7, 15])`: Explain specific concept indices.

Targets are shared across all input samples in a batch.

## Combining with Other Interpretations

Input-to-concept attributions complement other interpretation methods:

- **TopKInputs**: Identifies the most activating *samples* for each concept globally.
- **Concept attributions**: Reveals which *tokens* in a specific input drive each concept.
- **Concept output gradients**: Shows which concepts matter for each output class.

Together, they provide a complete interpretability story: which tokens activate which concepts,
and which concepts drive which predictions.

## API Reference

::: interpreto.concepts.base.ModelForInputsToConcepts
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - **init**
        - **call**

::: interpreto.attributions.base.InputsToConceptsAttributionsExplainer
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - process_inputs_to_explain_and_targets
        - post_processing
