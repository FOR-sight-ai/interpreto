---
icon: material/label-outline
---

# LLM Labels

`LLMLabels` uses a large language model to generate natural-language labels for each concept,
based on the top-k activating inputs. This provides a human-readable summary of what each concept represents.

As with `TopKInputs`, examples are whole samples for classification and pooled
text representations, and retained tokens for unpooled text representations.
Precomputed activations must match the selected mode. Context tokens around an
example (`k_context`) apply only to unpooled token-level interpretation;
`k_context` is forced to `0` for pooled representations, vocabulary examples,
and independently encoded words or n-grams.

## Labeling Model

Pass an existing `LLMInterface`, a Hugging Face repository ID, or a preloaded
`(model, tokenizer)` tuple through `llm_interface`:

```python
label_methods = LLMLabels(
    concept_explainer=concept_explainer,
    llm_interface="Qwen/Qwen3.5-9B",
)
```

When `llm_interface=None`, `LLMLabels` reuses the concept explainer's NNsight
splitter. This default is available only when that splitter is generation-capable.

## API Reference

::: interpreto.concepts.interpretations.LLMLabels
    handler: python
    options:
      show_root_heading: true
      show_source: true
      inherited_members: true
      members:
        - interpret

::: interpreto.commons.llm_interface.LLMInterface
    handler: python
    options:
      show_root_heading: true
      show_source: true
      inherited_members: true
      members:
        - generate
        - batch_generate
