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

`plot_lens` renders one compact row per model depth. The visible cells contain only the top prediction and use color
intensity to show relative confidence. Hover over a cell to inspect its numerical score and remaining top-k results.

::: interpreto.visualizations.plot_lens
