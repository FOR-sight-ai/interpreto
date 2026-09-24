# API: Lens

Lens methods decode an intermediate representation with the model's own prediction head. Interpreto provides:

- [Logit Lens](./methods/logit_lens.md), which applies the prediction head directly;
- [Tuned Lens](./methods/tuned_lens.md), which first learns an affine translator for the selected representation.

Both methods use the split point registered on
[`ModelWithSplitPoints`](../concepts/splitters/model_with_split_points.md). A wrapped model has one split point, so the returned dictionary has one entry named after that point.

## Choosing a split point

For a standard lens analysis, select the output of a transformer block: the post-block residual stream. For example, a GPT-2 block can be selected with `transformer.h.1`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from interpreto import LogitLens, ModelWithSplitPoints

model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_with_split_points = ModelWithSplitPoints(
    model,
    tokenizer=tokenizer,
    split_point="transformer.h.1",
)

lens = LogitLens(model_with_split_points, top_k=3)
explanations = lens.explain("Interpreto is useful.")
```

`ModelWithSplitPoints` can capture other modules, but their outputs do not necessarily represent the residual stream. For example, `transformer.h.1.mlp` is only the MLP update inside the block. Projecting such an activation may still be useful as a custom vocabulary-space analysis, but it should not be interpreted as the standard Logit Lens or Tuned Lens quantity. The activation must also have the hidden width expected by the projection path.

## Projection paths

Automatic projection is intentionally limited to Transformers layouts with a complete, module-backed suffix:

- language models with a recognizable language-model head and, when exposed separately, a final normalization module;
- single-label sequence classifiers with a composite classification head or a compatible model pooler followed by the classifier.

For a different layout, pass `head_name` and, when needed, `pre_head_name` explicitly. A bare vector classification head also needs a `pooling_strategy`, and decoder classifiers commonly need their final normalization module as `pre_head_name`. These arguments describe the suffix applied to the captured activation. An explicit path can define a useful custom projection, but it should only be described as reproducing the model classifier when it includes every required operation. Functional operations that are not represented by modules cannot be reconstructed by these arguments; choose another split point or projection path instead of treating an incomplete suffix as equivalent to the model classifier.

The supported task families are causal language modeling, masked language modeling, and single-label sequence classification. Regression, multi-label classification, and token classification are not supported. Compatibility still depends on the selected activation and the model's head layout, so an `AutoModel` family name alone does not guarantee a valid projection.

Some language models apply logit scaling or soft-capping as a functional operation after the reusable output modules. Lens methods reject configurations with an active transform of this kind rather than silently returning scores that disagree with the model's logits.

## Inputs and interpretation

Lens methods accept a string, a list of strings, or a tensor-backed `BatchEncoding`. Raw text is tokenized with the tokenizer attached to `ModelWithSplitPoints`. That tokenizer must already have a padding token or an end-of-sequence token; lens methods do not resize the model embeddings to add one. Tensor-backed batches must use right padding so the wrapped model can retain its native position-id behavior.

`top_scores` and the `mean_max_score` and `mean_target_score` metrics are softmax scores computed from intermediate logits. Intermediate states, especially early ones, do not follow the distribution seen by the final prediction head. The scores are therefore useful for rankings and within-experiment comparisons, but should not be presented as calibrated probabilities. A Tuned Lens reduces this distribution mismatch on its training distribution; it does not remove the need for held-out evaluation.

For masked language models, metrics without explicit labels measure visible, non-special token identity decodability. To evaluate masked-token predictions, pass a tensor-backed `BatchEncoding` together with the usual label tensor: original token ids at evaluated positions and `-100` elsewhere. Perplexity is reported only for causal next-token metrics.

The computation and notebook display steps can be kept separate:

```python
from interpreto import plot_lens

model_inputs = tokenizer(
    ["Interpreto is useful."],
    return_tensors="pt",
    padding=True,
    truncation=True,
)
results = lens.explain(model_inputs)
plot_lens(
    results,
    model_inputs,
    tokenizer=tokenizer,
    task=lens.task,
)
```

For sequence classification, pass readable class names with `label_names={...}` when displaying results.
