---
icon: material/code-json
---

# SplitterForGeneration

`SplitterForGeneration` is a `BaseSplitter` specialization for causal language models such as
`*ForCausalLM` and `*LMHeadModel` Hugging Face models. It keeps the split-point API explicit while
providing a simpler concept workflow for generation models than `ModelWithSplitPoints`.

## When to Use

Use `SplitterForGeneration` when:

- Your model is a Hugging Face causal language model.
- You want token-level activations at a single split point.
- You want flattened token activations for concept fitting or sample-wise token activations for interpretation.
- You do not need word, sentence, sample, or custom granularity aggregation.

Use `ModelWithSplitPoints` instead when you need the full `ActivationGranularity` API, including word,
sentence, or sample aggregation.

## Token Selection

By default, `get_activations` filters out padding and special tokens before returning activations. Pass
`include_special_tokens=True` to keep special tokens while still removing padding.

The return shape depends on `flatten_activations`:

- `flatten_activations=True` returns one tensor of shape `(n_tokens, hidden_dim)`.
- `flatten_activations=False` returns one tensor per input, each with shape `(n_tokens_i, hidden_dim)`.

## Quick Example

```python
from interpreto import SplitterForGeneration

split_model = SplitterForGeneration(
    "gpt2",
    split_point=10,
    batch_size=8,
    device_map="auto",
)

# Flattened token activations, suitable for concept fitting.
activations, _ = split_model.get_activations(texts, tqdm_bar=True)

# Sample-wise activations, useful when token alignment matters downstream.
sample_activations, _ = split_model.get_activations(
    texts,
    flatten_activations=False,
)
```

## Concept Gradients

For concept-to-output gradients, `SplitterForGeneration` reintegrates decoded concept activations at the
selected split point, then differentiates generation logits with respect to the concept activations. The
returned gradients are sample-wise tensors of shape `(n_targets, n_tokens_i, n_concepts)`.

## API Reference

::: interpreto.SplitterForGeneration
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - split_point
        - get_activations
        - get_latent_shape
