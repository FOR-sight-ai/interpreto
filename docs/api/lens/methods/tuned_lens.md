---
icon: material/tune-variant
---

# Tuned Lens

`TunedLens` follows
[Belrose et al. (2023)](https://arxiv.org/abs/2303.08112),
which learns a translator for each intermediate layer before applying the model prediction head.

An Interpreto `TunedLens` instance operates on the one split point registered on
[`ModelWithSplitPoints`](../../concepts/splitters/model_with_split_points.md), so it owns one affine `translator`. To study several layers, create one wrapped model and lens per split point. This keeps activation capture, training, and checkpoint metadata tied to the same model location.

For the standard method, choose a transformer block output such as `transformer.h.1` for GPT-2. An MLP or attention submodule output is only one update to the residual stream. A translator fitted on such an activation defines a different intervention and should not be reported as a standard Tuned Lens result.

`fit()` freezes the wrapped model and prediction path while training the translator to match the final model distribution. The `logit_lens` initialization starts from an identity map; `xavier` and the default `torch.nn.Linear` initialization are also available. Use separate training and evaluation inputs: fitting reduces distribution mismatch on the training distribution but does not make intermediate softmax scores calibrated probabilities.

The original method targets autoregressive language models. Interpreto also exposes the same distribution-matching objective for masked language models and compatible single-label sequence classifiers. Classification support depends on a projection path that faithfully represents the classifier suffix; see the [Lens overview](../overview.md) for automatic resolution and explicit projection arguments.

`save()` writes a tensor-only checkpoint with the declared model identity, split point, projection path, and translator state. `from_checkpoint()` checks that metadata before restoring the translator. The checkpoint does not hash the wrapped model's full parameter set, so applications that alter weights without changing the model's declared name or path must track that relationship themselves.

::: interpreto.TunedLens
    handler: python
    options:
      show_root_heading: true
      show_source: true
