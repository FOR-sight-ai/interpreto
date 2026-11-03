# API: Concept-based Methods

## Tutorials

- [Concept Explanations Examples](../../notebooks/classification_concept_tutorial.ipynb)

## Common API

The following abbreviated example shows the common workflow for concept-based
explainers. Replace the model, dataset, and concept-model parameters for your
use case.

```python
from interpreto import SplitterForClassification
from interpreto.concepts import ICAConcepts
from interpreto.concepts.interpretations import TopKInputs

# 1. Load and split the model.
splitter = SplitterForClassification("your_model_id", device_map="cuda")

# 2. Extract one latent representation per input.
activations, predictions = splitter.get_activations(dataset)

# 3. Fit a concept model.
concept_explainer = ICAConcepts(splitter, nb_concepts=50)
concept_explainer.fit(activations)

# 4. Interpret the concepts.
interpretation = TopKInputs(concept_explainer=concept_explainer).interpret(inputs=dataset)

# 5. Estimate concept contributions to model outputs.
concept_gradients = concept_explainer.concept_output_gradient(inputs=dataset)
```

## Step 1: Select a Splitter

> **For simple text classification:** Use [`SplitterForClassification`](./splitters/splitter_for_classification.md)
> It automatically detects the classification head of your classifier in most cases.
> Then, it considers the [CLS] token (the input of this head as the activations).
> Which means that there is one activation vector for each sample. (n, d)
> Thus `inputs_to_activations` is the encoder and `activations_to_outputs` is the head.

> **For other text models:** Use [`TextTokensSplitter`](./splitters/text_tokens_splitter.md)
> It can be used for any text model, but its main objective is generation models.
> Here you need to specify split point manually. It can, be the number of the layer
> or the name of the layer.
> It extracts one activation per retained token (by default, it filters out special tokens).
> It can pool these activations into one representation per sample with `token_pooling`.

`SplitterForGeneration` remains as a deprecated compatibility class for
`TextTokensSplitter(..., task="text-generation")`.
`ModelWithSplitPoints` has been removed; its compatibility symbol raises a
migration error directing callers to one of these supported splitters.

## Step 2: Extract Activations

Call the splitter's `get_activations()` method to build the activation dataset
used to fit a concept model.

!!! tip
    Larger and more representative activation datasets generally produce more
    useful concept spaces, but cost more to extract and fit.

!!! warning
    The source dataset has a strong effect on the concepts that are discovered.

## Step 3: Create a Concept Explainer

### Unsupervised Methods

Interpreto supports several unsupervised concept explainers through
`overcomplete`. See [SAEs](./concept_spaces/sae.md),
[Dictionary Learning](./concept_spaces/optim.md),
[Cockatiel](./concept_spaces/cockatiel.md), and
[Neurons as Concepts](./concept_spaces/neurons_as_concepts.md).

Their common parameters include:

- `splitter`: the model wrapper and selected split point.
- `nb_concepts`: the size of the concept space.
- `device`: the device used by trainable concept models.

### Supervised Methods (Probes)

Interpreto also supports supervised concept methods via [Probes](./probes.md).
Probes require labeled data (binary concept annotations) and learn a mapping from
activations to concept scores. They are useful when you already know which concepts
you want to test for.

```python
from interpreto.concepts import ProbeExplainer
from interpreto.concepts.probes import LinearRegressionProbe

probe = LinearRegressionProbe()
concept_explainer = ProbeExplainer(splitter, concept_model=probe)
concept_explainer.fit(activations, labels)
```

## Step 4: Fit the Concept Explainer

Fitting defines directions or regions in the model's latent space. Runtime
depends on the activation dataset, concept-space size, and method.

## Step 5: Interpret Concepts

### TopKInputs

[TopKInputs](./interpretations/topk_inputs.md) associates each concept with
its most activating examples. Classification and pooled text representations
use whole inputs; unpooled text representations use retained tokens.

```python
from interpreto.concepts.interpretations import TopKInputs

topk = TopKInputs(concept_explainer=concept_explainer, k=5)
topk_examples = topk.interpret(inputs=dataset, concepts_indices="all")
```

### LLMLabels

[LLMLabels](./interpretations/llm_labels.md) asks a language model to produce
natural-language labels from activating examples.

### Input-to-concept Attributions

[Concept Attributions](./interpretations/concept_attributions.md) identify
which tokens in an input activate a concept. They combine the concept pipeline
with perturbation-based attribution methods:

```python
from interpreto import Occlusion

# Get the bridge model that maps inputs → concept activations
explainer = Occlusion(
    concept_explainer.get_inputs_to_concepts_model(),
    splitter.tokenizer,
    batch_size=256,
)

# Explain all concepts (or pass targets for specific concepts)
results = explainer.explain(inputs)
```

Only perturbation-based methods (`Occlusion`, `Lime`, `KernelShap`, and
`Sobol`) support this bridge model.

## Step 6: Estimate Concept Contributions

`concept_output_gradient()` computes gradients from concept activations to
model outputs. Important parameters include:

- `targets`: class indices for classification or output positions for generation.
- `concepts_x_gradients`: multiply gradients by the concept activations when `True`. Defaults to `True`.
- `batch_size`: the requested model batch size; generation gradients execute sample by sample.

When `targets=None`, generation computes gradients for all real output
positions in each sample. Floating-point activations are normalized to the
concept model's dtype and device without detaching the gradient path.
