# API: Lens

This section groups the lens-based interpretability methods of Interpreto.

Current methods:

- [Logit Lens](./methods/logit_lens.md): project split activations through the model prediction head.
- [Tuned Lens](./methods/tuned_lens.md): learn one affine translator per split point before the same projection step.

Both methods are built around [`ModelWithSplitPoints`](../concepts/model_with_split_points.md) and accept the same high-level workflow:

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
    split_points="transformer.h.1.mlp",
)

lens = LogitLens(model_with_split_points, top_k=3)
explanations = lens.explain("Interpreto is useful.")
```

The generated class pages document the constructor arguments and examples in detail.
Both classes rely on [`ModelWithSplitPoints`](../concepts/model_with_split_points.md).
Raw text inputs are tokenized internally by the lens methods with the wrapped tokenizer.

For sequence classification, three projection cases are supported:

- a model-specific pooler or transform followed by a vector head
- a sequence-aware classification head that consumes 3D hidden states directly
- a bare vector head, which requires an explicit `pooling_strategy`

For notebook rendering, the computation and visualization steps can also be split explicitly:

```python
from interpreto.visualizations import display_lens_results

model_inputs = tokenizer(["Interpreto is useful."], return_tensors="pt", padding=True, truncation=True)
results = lens.explain(model_inputs)
display_lens_results(
    results,
    model_inputs,
    tokenizer=tokenizer,
    task=lens.task,
)
```

For sequence classification, readable class names should be passed explicitly through
`label_names={...}` when displaying the results.

The tiny `hf-internal-testing` checkpoints used in the documentation and tests are lightweight fixtures.
They are useful for quick checks, but they may expose uninitialized heads or meta-tensor loading paths
that are not representative of normal experimentation. The supported usage path for the lens methods
remains a fully loaded Hugging Face model or a fully materialized `ModelWithSplitPoints`.
