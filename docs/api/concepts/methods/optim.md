---
icon: material/middleware-outline
---

# Optimization-based Dictionary Learning

## Base abstract class

::: interpreto.concepts.methods.DictionaryLearningExplainer
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

## List of available methods

::: interpreto.concepts.ConvexNMFConcepts
    handler: python
    options:
      show_root_heading: true
      parameter_headings: false

::: interpreto.concepts.DictionaryLearningConcepts
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.ICAConcepts
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.KMeansConcepts
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.NMFConcepts
    handler: python
    options:
      show_root_heading: true
      parameter_headings: false
      filters:
        - "!^fit"

::: interpreto.concepts.PCAConcepts
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.SemiNMFConcepts
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.SparsePCAConcepts
    handler: python
    options:
      show_root_heading: true

::: interpreto.concepts.SVDConcepts
    handler: python
    options:
      show_root_heading: true
