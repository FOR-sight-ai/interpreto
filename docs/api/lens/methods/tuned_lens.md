---
icon: material/tune-variant
---

# Tuned Lens

`TunedLens` follows [Belrose et al. (2023)](https://arxiv.org/abs/2303.08112). It learns one affine residual
translator for each non-final model state and trains all of them to match the model's final prediction distribution.

The translators start at zero, making an unfitted Tuned Lens exactly equivalent to a Logit Lens. Fitting processes one
text at a time while projecting all model depths together. Use separate training and evaluation texts when assessing
the fitted lens.

::: interpreto.TunedLens
    handler: python
    options:
      show_root_heading: true
      show_source: true
