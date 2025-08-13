---
icon: material/middleware-outline
---

# Sparse Autoencoders (SAEs)

## List of available SAEs

::: interpreto.concepts.methods.VanillaSAEConcepts
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.methods.TopKSAEConcepts
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.methods.BatchTopKSAEConcepts
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.methods.JumpReLUSAEConcepts
    handler: python
    options:
      show_root_heading: true

## Abstract base class

::: interpreto.concepts.methods.SAEExplainer
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

## Loss Functions

These functions can be passed as the `criterion` argument in the `fit` method of the `SAEExplainer` class. `MSELoss` is the default loss function.

::: interpreto.concepts.methods.SAELossClasses
    handler: python
    options:
      show_root_heading: true
      show_attributes: true
      show_enums: true
