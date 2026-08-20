# Visualizations

## `plot_attributions`

::: interpreto.visualizations.plot_attributions

## `plot_concepts`

::: interpreto.visualizations.plot_concepts

## `plot_lens`

`plot_lens` renders the normalized output of `LogitLens.explain()` or `TunedLens.explain()` without running the model again. Pass the same tensor-backed `BatchEncoding` used for the explanation so that tokens and padding positions remain aligned. For sequence classification, `label_names` can map label ids to readable names. Like the other Interpreto plotting functions, it accepts `custom_css` and an optional `save_path`.

The displayed values are intermediate softmax scores, not calibrated probabilities. Language-model tooltips show the numerical top-k scores at each position. For classification, bar lengths show the corresponding class scores.

::: interpreto.visualizations.plot_lens
