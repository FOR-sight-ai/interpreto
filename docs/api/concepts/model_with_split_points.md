---
icon: material/code-json
---

`ModelWithSplitPoints` now uses the singular `split_point` argument/property because only one split point is supported.
The previous `split_points` argument/property remains temporarily available as a deprecated compatibility alias and emits a `DeprecationWarning` guiding users to `split_point`. It will be removed in version `0.6.0`.

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
