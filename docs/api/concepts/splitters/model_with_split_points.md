---
icon: material/code-json
---

`ModelWithSplitPoints` now uses the singular `split_point` argument/property because only one split point is supported.
The previous `split_points` argument/property remains temporarily available as a deprecated compatibility alias and emits a `DeprecationWarning` guiding users to `split_point`. It will be removed in version `1.0.0`.

## Specificities

In comparison to `SplitterForClassification` and `SplitterForGeneration`, the `ModelWithSplitPoints` class is more versatile.
It is more complex to use, but it covers the two other splitter cases.

In both `get_activations` and `_get_concept_output_gradients`, one needs to specify:

- `activation_granularity`: specifies which of the `(n, l, d)` activations to return.
It can be one of `CLS_TOKEN`, `ALL_TOKENS`, `TOKEN`, `WORD`, `SENTENCE`, or `SAMPLE`.
Use `activation_granularity=ModelWithSplitPoints.activation_granularities.TOKEN` to specify it.
- `aggregation_strategy`: how activations should be aggregated (only for `WORD`, `SENTENCE`, or `SAMPLE`).
It can be one of `SUM`, `MEAN`, `MAX`, or `SIGNED_MAX`.
Use `aggregation_strategy=ModelWithSplitPoints.aggregation_strategies.MEAN` to specify it.

::: interpreto.ModelWithSplitPoints
    handler: python
    options:
      show_root_heading: true
      show_attributes: true
      show_source: true
      members:
        - activation_granularities
        - aggregation_strategies
        - get_activations
        - get_latent_shape

::: interpreto.concepts.splitters.model_with_split_points.ActivationGranularity
    handler: python
    options:
      show_root_heading: true
      show_source: false
      show_if_no_docstring: true
      show_members: false

::: interpreto.concepts.splitters.model_with_split_points.GranularityAggregationStrategy
    handler: python
    options:
      show_root_heading: true
      show_source: false
      show_if_no_docstring: true
      show_members: false
