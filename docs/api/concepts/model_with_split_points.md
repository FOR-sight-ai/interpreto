---
icon: material/code-json
---

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
        - get_split_activations

::: interpreto.model_wrapping.model_with_split_points.ActivationGranularity
    handler: python
    options:
      show_root_heading: true
      show_source: false
      show_if_no_docstring: true
      show_members: false

::: interpreto.model_wrapping.model_with_split_points.GranularityAggregationStrategy
    handler: python
    options:
      show_root_heading: true
      show_source: false
      show_if_no_docstring: true
      show_members: false
