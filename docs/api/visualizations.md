# Visualizations

## `plot_attributions`

::: interpreto.visualizations.plot_attributions

## `plot_image_retrieval_attribution`

Display an `ImageRetrievalAttributionOutput` with the query above the retrieved
image, in a notebook or a self-contained HTML file:

```python
from interpreto import plot_image_retrieval_attribution

# retrieval_outputs is the list returned by your retrieval explainer.
plot_image_retrieval_attribution(retrieval_outputs[0], save_path="retrieval.html")
```

The displayed image is the raw input image (`input_image`, set by `explain`),
shown at its original aspect — not the processor's resized square. Heatmaps
live on the model grid and are warped onto the displayed image for rendering
only, using the same `resize_strategy` interpolation. This warp inverts
preprocessing exactly for resize-only processors (e.g. ColPali's); cropping or
padding processors would misalign the overlay. Without `input_image` (hand-built
outputs), the de-normalized model input is shown instead.

`max_image_side` (default 768, `None` for native resolution) downscales the
base image and overlays for display only; scores are never affected.

The default heatmap is a **similarity-weighted sum** of the token attribution
maps: each token map is first L1-normalized (absolute values summing to 1),
then weighted by that token's similarity, so low-similarity tokens barely
affect the default view. Hover a token to preview its own explanation; click
to pin it. Hovering another token temporarily overrides the pinned view. Click
the pinned token again, press Escape to return to the weighted sum. Keyboard
focus also previews tokens, and Enter/Space selects them.

Query highlights use the per-token **similarities**, max-normalized for display
so the strongest token always reaches full opacity regardless of how many
tokens share the query (the L1-normalized weights are only used to combine the
maps, never for display); the tooltip on each token shows its raw similarity.
Positive scores are red and negative scores blue by default.

Each displayed token-wise image map is normalized independently with a
zero-anchored min–max of absolute values, restoring signs afterward:

`sign(x) * abs(x) / max(abs(x))`

The resulting scores lie in [-1, 1], and opacity reflects their magnitude.
Constant nonzero maps remain visible, and all-zero maps stay transparent.
There is no legend or additional control panel; interaction is directly
through the tokens.

Patch explanations use **bicubic interpolation** for smooth transitions,
clamped to the original map's range before normalization to avoid interpolation
overshoot. Pass `resize_strategy=GranularityResizeStrategy.NEAREST` (imported
from `interpreto.commons.granularity`) for crisp patch boundaries. Pixel maps
are already at image resolution and are not interpolated.

The image is shown at model-input resolution, which may
differ from the original image if the processor cropped or resized it. PNG image
assets, CSS, and JavaScript are embedded, so saved HTML needs no network access.

::: interpreto.visualizations.plot_image_retrieval_attribution

## `plot_concepts`

::: interpreto.visualizations.plot_concepts
