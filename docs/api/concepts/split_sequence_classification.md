---
icon: material/code-json
---

# SplitSequenceClassification

`SplitSequenceClassification` is a specialized version of `ModelWithSplitPoints` designed for
`*ForSequenceClassification` HuggingFace models. It simplifies the setup by automatically identifying the
classification head as the split point and the granularity as the [CLS] token.

## When to Use

Use `SplitSequenceClassification` instead of `ModelWithSplitPoints` when:

- Your model is a Hugging Face `*ForSequenceClassification` model.
- You want to extract CLS-token activations without manually specifying a split point.
- You want a cleaner, faster concept pipeline for classification tasks.

## Additional Gain

It unlocks the inputs-to-concepts attributions workflow, which is not possible with `ModelWithSplitPoints`.

## Quick Example

```python
from interpreto import SplitSequenceClassification

split_model = SplitSequenceClassification(
    "nateraw/bert-base-uncased-emotion",
    batch_size=32,
    device_map="cuda",
)

# Compute activations on a dataset
activations = split_model.get_activations(texts, tqdm_bar=True)
```

## API Reference

::: interpreto.SplitSequenceClassification
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - classification_head_name
        - inputs_to_activations
        - activations_to_outputs
        - get_activations
        - get_split_activations
        - get_latent_shape
