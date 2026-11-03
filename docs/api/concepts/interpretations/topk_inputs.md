---
icon: material/keyboard-tab-reverse
---

# TopKInputs

`TopKInputs` identifies the most activating inputs for each concept globally.
It provides a **global** interpretation by finding which elements in your dataset best characterize each concept direction.

Display units follow the splitter and pooling mode:

- Classification uses whole input samples because each input has one representation.
- Text splitting without pooling uses individual retained tokens.
- Text splitting with `token_pooling` uses whole input samples.
- `use_vocab=True` ranks independently encoded vocabulary entries.
- `use_unique_words=True` ranks independently encoded words or n-grams and requires a pooled representation.

Precomputed `latent_activations` or `concepts_activations` must match that mode.
Pooling is not inferred from tensor shapes.

## Quick Example

```python
from interpreto.concepts.interpretations import TopKInputs

topk = TopKInputs(
    concept_explainer=concept_explainer,
    k=5,
)

topk_examples = topk.interpret(inputs=dataset, concepts_indices="all")
```

For pooled text representations, use the same pooling mode for extraction and interpretation:

```python
activations, _ = splitter.get_activations(dataset, token_pooling="mean")
topk = TopKInputs(concept_explainer=concept_explainer, token_pooling="mean", k=5)
topk_inputs = topk.interpret(
    inputs=dataset,
    latent_activations=activations,
    concepts_indices="all",
)
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
