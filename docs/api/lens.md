# Lens

Lens methods decode the residual stream at every transformer block boundary. They use an
[`AllLayersSplitter`](concepts/splitters/all_layers_splitter.md) explicitly so the same configured model, tokenizer,
and device can be reused.

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
from interpreto import AllLayersSplitter, TunedLens, plot_lens

text = "Paris is the capital of"
splitter = AllLayersSplitter("distilgpt2")
lens = TunedLens(splitter, top_k=3)
lens.fit(["Paris is the capital of France.", "Rome is the capital of Italy."], epochs=1)
results = lens(text)

plot_lens(results, text, tokenizer=splitter.tokenizer)
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

`plot_lens` renders each layer as a table. Language-model rows show the input token and numerical top-k predictions;
classification rows show class names and scores.

::: interpreto.visualizations.plot_lens
