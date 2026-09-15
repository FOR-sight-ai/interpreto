---
icon: material/layers-triple-outline
---

# Logit Lens

`LogitLens` follows the method introduced by
[nostalgebraist](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). It projects the residual
stream at every model depth through the model's native prediction path. The method has no learned parameters.

The prediction head was trained on final states, so early-layer scores should be interpreted as rankings rather than
calibrated probabilities. Tuned Lens addresses part of this distribution mismatch by learning a translator for every
non-final state.

::: interpreto.LogitLens
    handler: python
    options:
      show_root_heading: true
      show_source: true
