---
icon: material/layers-triple-outline
---

# Logit Lens

`LogitLens` follows the original idea introduced by
[nostalgebraist](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
and is related to the vocabulary-space analysis studied by
[Geva et al. (2022)](https://aclanthology.org/2022.emnlp-main.3/).

The method applies the configured prediction path directly to an intermediate state. For a language model, that path normally consists of the final normalization and language-model head. `LogitLens` has no learned parameters. In Interpreto, the state comes from the split point registered on
[`ModelWithSplitPoints`](../../concepts/splitters/model_with_split_points.md).

Use a transformer block output, such as `transformer.h.1` for GPT-2, when running a standard Logit Lens analysis. A submodule output such as `transformer.h.1.mlp` is an update to the residual stream rather than the residual stream itself. Interpreto can project a shape-compatible submodule output, but the result is then a custom activation projection and should be described as such.

The prediction head was trained on final-layer states, not on intermediate states. Consequently, the returned softmax values are scores for ranking intermediate predictions, not calibrated probabilities. This distribution-shift limitation is strongest at early layers and should be considered when comparing layers or models.

Automatic head resolution covers common Hugging Face language-model layouts and sequence classifiers with a complete module-backed suffix. For another architecture, provide an explicit compatible `head_name`, `pre_head_name`, and, for a bare vector classifier, `pooling_strategy`. See the [Lens overview](../overview.md) for the projection contract and task limitations.

::: interpreto.LogitLens
    handler: python
    options:
      show_root_heading: true
      show_source: true
