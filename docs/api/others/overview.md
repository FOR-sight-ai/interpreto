# API: Other Interpretability Methods

This section contains interpretability methods that provide unique insights into model behavior but don't fit into the traditional attribution or concept-based categories.

## Common API

```python
from interpreto.model_wrapping import ModelWithSplitPoints
from interpreto.others import LogitLens  # or specific implementations

# 1. Load and wrap your model
model_with_split_points = ModelWithSplitPoints(
    "your_model_id",
    split_points="your_split_points",
    model_autoclass="your_model_autoclass",
    device_map="auto"
)

# 2. Instantiate the interpretability method
lens = LogitLens(model_with_split_points, tokenizer, **kwargs)

# 3. Generate explanations
explanations = lens(inputs, layers_name=None)

# 4. Visualize results (method-dependent)
lens.lens(inputs, layers_name=None)  # For interactive visualization
```

The API typically follows these steps:

### Step 1: Model Preparation with `ModelWithSplitPoints`

Similar to other interpreto methods, you need to wrap your model with `ModelWithSplitPoints` to access intermediate activations. This wrapper allows you to:

- Extract activations from specific layers
- Process different model types uniformly
- Handle batching efficiently

### Step 2: Method Instantiation

Each method in the "others" category has its own initialization parameters, but they generally follow a common pattern:

- `model` (`ModelWithSplitPoints`): The wrapped model to analyze
- `tokenizer` (`PreTrainedTokenizer`): The tokenizer associated with the model
- `batch_size` (`int`): Batch size for processing
- Method-specific parameters vary by implementation

### Step 3: Generate Explanations

Methods typically provide a `__call__` method or `explain` method that:

- Takes input text, tokenized inputs, or tensor inputs
- Optionally specifies which layers to analyze
- Returns structured explanations

### Step 4: Visualization

Most methods provide specialized visualization capabilities:

- Interactive HTML visualizations for detailed exploration
- Summary visualizations for quick insights
- Export capabilities for further analysis

## Available Methods

➡️ **Logit Lens Methods:**

- [General Logit Lens](./methods/logit_lens.md): Adaptive logit lens implementation that works with both language models and classification models, showing intermediate predictions at each layer.
