---
icon: material/layers-triple-outline
---

# Logit Lens

Implementation of the Logit Lens technique from [Interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) by nostalgebraist. This mechanistic interpretability method analyzes what a transformer model "thinks" at each layer by projecting intermediate activations through the model's final prediction head.

## Main Classes

::: interpreto.others.logit_lens_general.LogitLens
    handler: python
    options:
      show_root_heading: true
      show_source: true

## Base Implementation

::: interpreto.others.logit_lens_general.BaseLogitLens
    handler: python
    options:
      show_root_heading: true
      show_source: true
      inherited_members: true
      members:
        - do_lens
        - explain
        - __call__

## Language Model Implementation

::: interpreto.others.logit_lens_general.LanguageModelLogitLens
    handler: python
    options:
      show_root_heading: true
      show_source: true
      inherited_members: true
      members:
        - do_lens
        - lens
        - visualize_logit_lens_interactive

## Classification Model Implementation

::: interpreto.others.logit_lens_general.ClassificationLogitLens
    handler: python
    options:
      show_root_heading: true
      show_source: true
      inherited_members: true
      members:
        - do_lens
        - lens
        - set_num_classes
        - visualize_classification_lens
