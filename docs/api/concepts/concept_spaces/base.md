---
icon: material/middleware-outline
---

# Base Classes

::: interpreto.concepts.ConceptEncoderExplainer
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - fit
        - verify_activations
        - interpret
        - get_inputs_to_concepts_model

::: interpreto.concepts.ConceptAutoEncoderExplainer
    handler: python
    options:
      show_root_heading: true
      show_source: true
      inherited_members: true
      members:
        - fit
        - encode_activations
        - decode_concepts
        - get_dictionary
        - interpret
        - concept_output_gradient
