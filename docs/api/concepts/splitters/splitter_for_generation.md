---
icon: material/code-json
---

# SplitterForGeneration

!!! warning "Deprecated"
    `SplitterForGeneration` is deprecated. Use
    [`TextTokensSplitter`](./text_tokens_splitter.md) with
    `task="text-generation"` instead. The compatibility class preserves the
    historical constructor signature and always selects the text-generation task.

## API Reference

::: interpreto.SplitterForGeneration
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - split_point
        - get_activations
        - get_latent_shape
