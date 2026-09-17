---
icon: material/code-json
---

# SplitterForClassification

`SplitterForClassification` is a specialized version of `BaseSplitter` designed for
`*ForSequenceClassification` HuggingFace models. It simplifies the setup by automatically identifying the
classification head as the split point and exposing the single representation consumed by that head as activations:
an already-pooled vector, the first sequence token (for example, RoBERTa), or the last
non-padding token for decoder-style token-wise `score` heads (for example, Llama/GPT classifiers).

## When to Use

Use `SplitterForClassification` when:

- Your model is a Hugging Face `*ForSequenceClassification` model.
- You want to extract classification representations without manually specifying a split point.
- You want a cleaner, faster concept pipeline for classification tasks.

## Additional Gain

It also supports the inputs-to-concepts attribution workflow.

## Quick Example

```python
from interpreto import SplitterForClassification

splitter = SplitterForClassification(
    "nateraw/bert-base-uncased-emotion",
    batch_size=32,
    device_map="cuda",
)

# Compute activations and predictions on a dataset
activations, predictions = splitter.get_activations(texts, tqdm_bar=True)
```

## API Reference

::: interpreto.SplitterForClassification
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - split_point
        - inputs_to_activations
        - activations_to_outputs
        - get_activations
        - get_latent_shape
