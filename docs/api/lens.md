# Lens

Lens methods decode the residual stream at every transformer block boundary. Pass a model repository ID directly for
the common case, or provide a configured [`AllLayersSplitter`](concepts/splitters/all_layers_splitter.md) when you need
to select a model class, tokenizer, device, or transformer block path.

Inference accepts one text at a time. To explain several texts, iterate over them. `TunedLens.fit()` accepts either one
text or an iterable and trains on each text sequentially.

## Classification with Logit Lens

```python
from transformers import AutoModelForSequenceClassification

from interpreto import AllLayersSplitter, LogitLens, plot_lens

text = "Interpreto makes model decisions easier to inspect."
splitter = AllLayersSplitter(
    "distilbert-base-uncased-finetuned-sst-2-english",
    automodel=AutoModelForSequenceClassification,
)
lens = LogitLens(splitter, top_k=2)
results = lens(text)

plot_lens(results, text, tokenizer=splitter.tokenizer, label_names=["negative", "positive"])
```

## Generation with Tuned Lens

```python
from interpreto import TunedLens, plot_lens

text = "Paris is the capital of"
lens = TunedLens("distilgpt2", top_k=3)
lens.fit(["Paris is the capital of France.", "Rome is the capital of Italy."], epochs=1)
results = lens(text)

plot_lens(results, text, tokenizer=lens.splitter.tokenizer)
```

For causal language models, predictions are aligned with their observed target tokens by default. Each column therefore
shows a target token, while the bottom `Input` row shows the preceding token that produced its prediction. Correct top
predictions are outlined in green.

`generate()` uses the same convention. Include the prompt when plotting so the visualization can display the input to
the first generated prediction:

```python
prompt = "Although the committee initially rejected the proposal,"
generated_text, generated_results = lens.generate(prompt, max_new_tokens=10)

plot_lens(generated_results, prompt + generated_text, tokenizer=lens.splitter.tokenizer)
```

Set `align=False` on both calls for the conventional next-token view, where each column contains the prediction made
after its displayed input token:

```python
generated_text, generated_results = lens.generate(prompt, max_new_tokens=10, align=False)
plot_lens(
    generated_results,
    prompt + generated_text,
    tokenizer=lens.splitter.tokenizer,
    align=False,
)
```

## `LogitLens`

::: interpreto.LogitLens
    handler: python
    options:
      show_root_heading: true
      show_source: true

## `TunedLens`

::: interpreto.TunedLens
    handler: python
    options:
      show_root_heading: true
      show_source: true

## `plot_lens`

`plot_lens` renders model outputs from the final layer down to the embeddings, followed by the actual input tokens. The
`Embeddings` row is the prediction head applied before the first transformer block; it is not the raw model input. The
visible cells contain only the top prediction and use color intensity to show relative confidence. A green outline marks
a correct prediction. Hover over a cell to inspect its numerical score and remaining top-k results.

::: interpreto.visualizations.plot_lens
