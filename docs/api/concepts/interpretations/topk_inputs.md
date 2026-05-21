---
icon: material/keyboard-tab-reverse
---

# TopKInputs

`TopKInputs` identifies the most activating inputs (tokens, words, sentences, or samples) for each concept globally.
It provides a **global** interpretation by finding which elements in your dataset best characterize each concept direction.

## Quick Example

```python
from interpreto.concepts.interpretations import TopKInputs
from interpreto.model_wrapping.model_with_split_points import ActivationGranularity

topk = TopKInputs(
    concept_explainer=concept_explainer,
    k=5,
    activation_granularity=ActivationGranularity.CLS_TOKEN,
    use_unique_words=3,  # consider all 3-grams as unique words
)

topk_words = topk.interpret(inputs=dataset, concepts_indices="all")
```

## API Reference

::: interpreto.concepts.interpretations.TopKInputs
    handler: python
    options:
      show_root_heading: true
      show_source: true
      inherited_members: true
      members:
        - interpret

::: interpreto.concepts.interpretations.extract_ngrams
    handler: python
    options:
      show_root_heading: true
      show_source: true
