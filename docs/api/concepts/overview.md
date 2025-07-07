# API: Concept-based Methods

Checkout a [concrete example notebook](https://github.com/FOR-sight-ai/interpreto/blob/main/docs/notebooks/concept_examples.ipynb) to see how to use the concept-based explainers.

## Common API

```python
from interpreto import ModelWithSplitPoints
from interpreto.concepts import ICAConcepts  # Many other possibilities here
from interpreto.concepts.interpretations import TopKInputs

# 1. Load and split your model
model_with_split_points = ModelWithSplitPoints(
    "your_model_id",
    split_points="your_split_point",
    model_autoclass="your_model_autoclass",
    nb_concepts=50,
    device_map="cuda"
)

# 2. Compute the model activations on the split_point
activations = model_with_split_points.get_activations(dataset)

# 3. Instantiate the concept-based explainer
concept_explainer = ICAConcepts(model_with_split_points)

# 4. Fit the concept-based explainer on the activations
concept_explainer.fit(activations)

# 5. Interpret the obtained concepts
interpretation = concept_explainer.interpret(TopKInputs, inputs=dataset)
```

The API has five steps for now but will have 6 in the future:

### Step 1: Load and split your model with `ModelWithSplitPoints`

`ModelWithSplitPoints` is a wrapper around your model to split it into different parts.
It allows to obtain activations on a specific split point.
The main method is thus `get_activations` which takes inputs and returns a dictionary of activations.

It can be initialized from the model repo id or from a model instance.

More details in the [`ModelWithSplitPoints` documentation](./model_with_split_points.md).

### Step 2: Compute the model activations on the split_point

At this step, there are two important parameters:

- `activation_granularity`: specifies which of the `(n, l, d)` activations to return.
It can be one of `ALL`, `CLS_TOKEN`, `ALL_TOKENS`, `TOKEN`, `WORD`, `SENTENCE`, `SAMPLE`.
Use `activation_granularity=ModelWithSplitPoints.activation_granularities.TOKEN` to specify it.

- `aggregation_strategy`: the mode used for inference.
It can be one of `SUM`, `MEAN`, `MAX`, or `SIGNED_MAX`.
Use `aggregation_strategy=ModelWithSplitPoints.aggregation_strategies.MEAN` to specify it.

### Step 3: Instantiate the concept-based explainer

The concept-based explainer is an object use to define the concept space.
Interpreto supports many concept-based explainers by wrapping over `overcomplete`.
See the following documentation for more details: [SAEs](./methods/sae.md);
[Dictionary Learning](./methods/optim.md); [Cockatiel](./methods/cockatiel.md); and [Neurons as concepts](.methods/neurons_as_concepts.md).

They has few key parameters:

- `model_with_split_points`: the model with split points.
- `nb_concepts`: the number of concepts to use.
- `device`: the device for training and inference for SAEs

### Step 4: Fit the concept-based explainer on the activations

The goal is to define the concept space. Each concept correspond to a direction or polytope in the latent space.
This may take quite a long time depending on the number of concepts and the size of the activations.

### Step 5: Interpret the obtained concepts

After the fit, concepts are abstract direction in the middle of the model.
To interpret the concepts, we need to communicate what they correspond to to the user.
For now, the only method is [TopKInputs](./interpretations/topk_inputs.md),
which associate each concept to the topk inputs that activates it the most.
These inputs can be tokens, words, sentences, or samples.

### Step 6: [Not yet implemented] Evaluate concepts' contributions to the output

There are often a lot of concepts, but only few are contributing to the output.
This is often less than the number of activated concepts.
To do so, we apply contribution methods from the concept space to the model output.
