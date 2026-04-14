---
icon: material/layers-triple-outline
---

# Logit Lens

`LogitLens` follows the original idea introduced by
[nostalgebraist](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
and is related to the vocabulary-space analysis studied by
[Geva et al. (2022)](https://aclanthology.org/2022.emnlp-main.3/).

In Interpreto, the method is exposed as a single user-facing class built on top of
[`ModelWithSplitPoints`](../../concepts/model_with_split_points.md).

Constructor arguments and examples are documented directly on the generated class page below.
When possible, prefer building the lens from a fully loaded Hugging Face model wrapped by
[`ModelWithSplitPoints`](../../concepts/model_with_split_points.md). Tiny
`hf-internal-testing` checkpoints are convenient fixtures, but they can expose meta-tensor
loading paths that do not reflect the usual experimental workflow.
Raw text inputs are tokenized internally by the lens methods with the wrapped tokenizer.

::: interpreto.LogitLens
    handler: python
    options:
      show_root_heading: true
      show_source: true
