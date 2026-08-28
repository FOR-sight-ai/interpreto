# API: Concept-based Methods

## Tutorials

- [Concept Explanations Examples](../../notebooks/classification_concept_tutorial.ipynb)

## Common API

This code snippet will not work in practice as some of the parameters are not defined.
Nonetheless, it shows the common API for all concept-based explainers.

```python
from interpreto import ModelWithSplitPoints
from interpreto.concepts import ICAConcepts  # Many other possibilities here
from interpreto.concepts.interpretations import TopKInputs

# 1. Load and split your model
model_with_split_points = ModelWithSplitPoints(
    "your_model_id", split_point="your_split_point", automodel="your_automodel", nb_concepts=50, device_map="cuda"
)

# 2. Compute the model activations on the split_point
activations, predictions = model_with_split_points.get_activations(dataset)

# 3. Instantiate and fit the concept-based explainer on the activations
concept_explainer = ICAConcepts(model_with_split_points)
concept_explainer.fit(activations)

# 4. Interpret the obtained concepts
interpretation = TopKInputs(concept_explainer).interpret(dataset)

# 5. Evaluate concepts' contributions to the output
concept_gradients = concept_explainer.concept_output_gradient(inputs=dataset)
```

The API has five steps:

### Step 1: Load and split your model with a splitter

Determine from which point of your model the activations should be extracted.

There are three splitters depending on you use-case:

> **For classification models:** Use [`SplitterForClassification`](./splitters/splitter_for_classification.md)
> It automatically detects the classification head of your classifier in most cases.
> Then, it considers the [CLS] token (the input of this head as the activations).
> Which means that there is one activation vector for each sample. (n, d)
> Thus `inputs_to_activations` is the encoder and `activations_to_outputs` is the head.

> **For causal language models:** Use [`SplitterForGeneration`](./splitters/splitter_for_generation.md)
> Here you need to specify split point manually. It can, be the number of the layer
> or the name of the layer.
> Then we consider each token latent activations as activations. (n * l, d).

> **For more complex cases:** Use [`ModelWithSplitPoints`](./splitters/model_with_split_points.md)
> It is more versatile, but also more complex.
> One needs to specify a split point manually and consider the granularity (see below).
> Thus it covers the two other splitter cases but less optimally.
> This can be useful for word-level probing for example.
> Or to split classifiers elsewhere.

### Step 2: Compute the model activations on the split_point

This step rely on the `.get_activations` method of the splitter.
The idea is to compute a dataset of activations.
To latter fit the concept model on.

!!! tip
    The larger the activations dataset, the more accurate the concept model will be.
    However, the more expensive it is to compute.

!!! warning
    The dataset used has a huge impact on the resulting concepts.

### Step 3: Instantiate the concept-based explainer

The concept-based explainer is an object use to define the concept space.

#### Unsupervised methods

Interpreto supports many **unsupervised** concept-based explainers by wrapping over `overcomplete`.
See the following documentation for more details: [SAEs](./concept_spaces/sae.md);
[Dictionary Learning](./concept_spaces/optim.md); [Cockatiel](./concept_spaces/cockatiel.md); and [Neurons as concepts](./concept_spaces/neurons_as_concepts.md).

They has few key parameters:

- `model_with_split_points`: the model wrapper with a split point.
- `nb_concepts`: the number of concepts to use.
- `device`: the device for training and inference for SAEs

#### Supervised methods (Probes)

Interpreto also supports **supervised** concept methods via [Probes](./probes.md).
Probes require labeled data (binary concept annotations) and learn a mapping from
activations to concept scores. They are useful when you already know which concepts
you want to test for.

```python
from interpreto.concepts import ProbeExplainer
from interpreto.concepts.probes import LinearRegressionProbe

probe = LinearRegressionProbe(nb_concepts=3, input_size=768)
concept_explainer = ProbeExplainer(model_with_split_points, concept_model=probe)
concept_explainer.fit(activations, y=labels)
```

See the [Probes (Supervised)](./probes.md) documentation for the full list of available probes.

### Step 4: Fit the concept-based explainer on the activations

The goal is to define the concept space. Each concept correspond to a direction or polytope in the latent space.
This may take quite a long time depending on the number of concepts and the size of the activations.

### Step 5: Interpret the obtained concepts (unsupervised methods only)

After the fit, concepts are abstract directions in the middle of the model.
To interpret the concepts, we need to communicate what they correspond to to the user.
There are several interpretation methods available:

#### TopKInputs (global interpretation)

[TopKInputs](./interpretations/topk_inputs.md) associates each concept to the top-k inputs that activate it the most.
These inputs can be tokens, words, sentences, or samples.

```python
from interpreto.concepts.interpretations import TopKInputs

topk = TopKInputs(concept_explainer, k=5)
topk_words = topk.interpret(inputs=dataset, concepts_indices="all")
```

#### LLM Labels

[LLM Labels](./interpretations/llm_labels.md) uses a large language model to generate natural-language labels
for each concept based on its top-k activating inputs.

#### Input-to-concept attributions (local interpretation)

[Concept Attributions](./interpretations/concept_attributions.md) reveal which **tokens in a specific input**
are responsible for activating a given concept. This provides a **local** (per-sample) interpretation,
complementing the global view offered by TopKInputs.

This is done by combining the concept framework with perturbation-based attribution methods:

```python
from interpreto import Occlusion

# Get the bridge model that maps inputs → concept activations
explainer = Occlusion(
    concept_explainer.get_inputs_to_concepts_model(),
    model_with_split_points.tokenizer,
    batch_size=256,
)

# Explain all concepts (or pass targets=torch.arange(5) for specific concepts)
results = explainer.explain(inputs)
```

> **Note:** Only perturbation-based methods (Occlusion, Lime, KernelShap, Sobol) are supported.
> Gradient-based methods are not compatible with the input-to-concept pipeline.

For classification models, use [`SplitterForClassification`](./splitters/splitter_for_classification.md)
instead of `ModelWithSplitPoints` for a simpler setup.

More details in the [Concept Attributions documentation](./interpretations/concept_attributions.md).

### Step 6: Evaluate concepts' contributions to the output (unsupervised methods only)

There are often a lot of concepts, but only few are contributing to the output.
This is often less than the number of activated concepts.
To do so, we apply contribution methods from the concept space to the model output.

For now only the gradient of the concept to output function is implemented.
Use the `concept_output_gradient` method to compute the gradient of the concept to output function.

There are few important parameters:

- `targets`: specify which outputs of the model should be used to compute the gradients.
- `activation_granularity`: use the same as step 2.
- `aggregation_strategy`: use the same as step 2.
- `concepts_x_gradients`: set to `True` to multiply the gradient by the concepts activations.
- `batch_size`: Batch size for the model. As this function computes gradients, it will be smaller than the batch size used in the `get_activations` method.
