---
icon: material/middleware-outline
---

# :parrot: Cockatiel

Implementation of the COCKATIEL framework from [COCKATIEL: COntinuous Concept ranKed ATtribution with Interpretable ELements for explaining neural net classifiers on NLP](https://aclanthology.org/2023.findings-acl.317/) by Jourdan et al. (2023).

::: interpreto.concepts.Cockatiel
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
