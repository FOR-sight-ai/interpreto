# API: Lens

Lens methods decode the residual stream at every transformer block boundary:

- [Logit Lens](./methods/logit_lens.md) applies the model prediction head directly.
- [Tuned Lens](./methods/tuned_lens.md) first applies one learned affine translator per depth.

Both methods use [`AllLayersSplitter`](../concepts/splitters/all_layers_splitter.md). It captures the state before the
first block and after every block in one trace. The lens concatenates those states along the leading dimension and
passes them through the model's native prediction path together.

```python
from interpreto import AllLayersSplitter, LogitLens

splitter = AllLayersSplitter("hf-internal-testing/tiny-random-gpt2")
lens = LogitLens(splitter, top_k=3)
results = lens("Interpreto is useful.")
```

`results` is ordered like `splitter.activation_names`. Each entry contains `top_indices` and `top_scores`. Language
model outputs have shape `(1, sequence_length, k)`; sequence-classification outputs have shape `(1, k)`, where `k` is
`top_k` capped at the model's output size. The singleton dimension represents the one input text. Input batches and
pre-tokenized inputs are intentionally not part of this initial API.

The scores are normalized over the complete output vocabulary or class set. Intermediate states, particularly early
ones, do not follow the final prediction head's training distribution, so these values are useful for rankings and
within-model comparisons rather than as calibrated probabilities.

## Tuned Lens

A Tuned Lens contains one translator for every non-final state. All translators start at zero and are applied
residually, so a new Tuned Lens initially returns the same predictions as a Logit Lens. `fit()` trains them jointly by
minimizing their KL divergence to the model's final prediction distribution.

```python
from interpreto import TunedLens

tuned_lens = TunedLens(splitter, top_k=3)
losses = tuned_lens.fit(
    ["Interpreto is useful.", "Lens methods expose intermediate predictions."],
    epochs=1,
)
results = tuned_lens("Interpreto is useful.")
```

`TunedLens` is a regular `torch.nn.Module`; use its `state_dict()` when saving trained translators.

## Visualization

```python
from interpreto import plot_lens

plot_lens(results, "Interpreto is useful.", tokenizer=splitter.tokenizer)
```

For sequence classification, pass `label_names` to display readable class names.
