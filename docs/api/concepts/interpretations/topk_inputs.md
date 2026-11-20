---
icon: material/keyboard-tab-reverse
---

# TopKInputs

`TopKInputs` identifies the most activating inputs for each concept globally.
It provides a **global** interpretation by finding which elements in your dataset best characterize each concept direction.

Display units follow the splitter and pooling mode:

- Classification ranks whole input samples by default because each input has one representation.
- Text splitting without pooling uses individual retained tokens.
- A `TextTokensSplitter` with `token_pooling` set to a pooling mode uses whole input samples.
- `use_vocab=True` ranks independently encoded vocabulary entries.
- With one representation per input, `use_unique_words=True` extracts unique words from the dataset and encodes each
  word as a separate input before ranking. Pass an integer such as `use_unique_words=3` to include up to 3-grams.
  This does not aggregate token activations into word-level activations from the original samples.

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

Same API for pooled representations:

```python
from interpreto import TextTokensSplitter
from interpreto.concepts import NeuronsAsConcepts, TopKInputs

splitter = TextTokensSplitter("gpt2", split_point=5, token_pooling="mean")
concept_explainer = NeuronsAsConcepts(splitter)
activations, _ = splitter.get_activations(dataset)
topk = TopKInputs(concept_explainer=concept_explainer, k=5)
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
