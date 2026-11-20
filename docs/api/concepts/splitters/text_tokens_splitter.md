---
icon: material/code-json
---

# TextTokensSplitter

`TextTokensSplitter` extracts hidden representations from a user-selected layer
of a causal or encoder text model. Select the NNsight loading task explicitly:

- `task="text-generation"` for causal language models.
- `task="feature-extraction"` for encoder models.

By default, padding and special tokens are removed and the retained token
representations are flattened into a tensor of shape `(n_tokens, hidden_dim)`.
Set `include_special_tokens=True` to retain special tokens, or
`flatten_activations=False` to return one tensor per input.

## Pooling

Set `token_pooling` when creating the splitter to obtain one representation per
input. Supported values are `"mean"`, `"max"`, `"min"`, `"signed_max"`, `"first"`,
and `"last"`. The default, `None`, keeps one representation per retained token.
Padding is always excluded, and special tokens are included only when
`include_special_tokens=True`.

```python
from interpreto import TextTokensSplitter

splitter = TextTokensSplitter(
    "gpt2",
    split_point=10,
    task="text-generation",
    # token_pooling=None,  # default, flattened representations, one per token (n * l, d)
    token_pooling="mean",  # one representation per sample (n, d)
    batch_size=8,
    device_map="auto",
)

sample_activations, _ = splitter.get_activations(texts)
```

Concept-to-output gradients are available only with
`task="text-generation"`. They are computed sample by sample at token level,
regardless of `token_pooling`, and returned as one tensor per input with shape
`(n_targets, n_tokens, n_concepts)`.

## API Reference

::: interpreto.TextTokensSplitter
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - split_point
        - inputs_to_activations
        - get_activations
        - get_latent_shape
