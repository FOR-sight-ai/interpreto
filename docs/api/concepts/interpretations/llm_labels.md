---
icon: material/label-outline
---

# LLM Labels

`LLMLabels` uses a large language model to generate natural-language labels for each concept,
based on the top-k activating inputs. This provides a human-readable summary of what each concept represents.

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
