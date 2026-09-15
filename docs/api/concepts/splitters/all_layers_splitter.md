# All Layers Splitter

`AllLayersSplitter` captures the residual stream before the first transformer block and after every block. It also
applies the wrapped model's native normalization, pooling, and prediction head to one or more residual states.

::: interpreto.AllLayersSplitter
    handler: python
    options:
      show_root_heading: true
      show_source: true
