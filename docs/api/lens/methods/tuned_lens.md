---
icon: material/tune-variant
---

# Tuned Lens

`TunedLens` follows
[Belrose et al. (2023)](https://arxiv.org/abs/2303.08112),
which learns a translator for each intermediate layer before applying the model prediction head.

The original paper focuses on autoregressive language models.
Interpreto extends the same projection pipeline to sequence classification through
[`ModelWithSplitPoints`](../../concepts/model_with_split_points.md).

Constructor arguments and examples are documented directly on the generated class page below.
When possible, prefer building the lens from a fully loaded Hugging Face model wrapped by
[`ModelWithSplitPoints`](../../concepts/model_with_split_points.md). Tiny
`hf-internal-testing` checkpoints are convenient fixtures, but they can expose meta-tensor
loading paths that do not reflect the usual experimental workflow.
Raw text inputs are tokenized internally by the lens methods with the wrapped tokenizer.

::: interpreto.TunedLens
    handler: python
    options:
      show_root_heading: true
      show_source: true
