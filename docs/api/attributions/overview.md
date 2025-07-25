# API: Attributions Methods

## Common API

```
from interpreto import Method

explainer = Method(model, tokenizer, kwargs)
explanation = explainer(inputs, targets)
```

The API have two steps:

**Step 1:** Explainer instantiation: `Method` is a generic name for an attribution method from the list of available methods (below). It inherits from the `AttributionExplainer` base class. Its initialization takes the general parameters explained here (and parameters that are specific to each method, explained in their own documentation):

- `model` (`PreTrainedModel`): Hugging Face model to explain,
- `tokenizer` (`PreTrainedTokenizer`): Hugging Face tokenizer associated with the model,
- `batch_size` (`int`): batch size for the attribution method,
- `granularity` (`Granularity`): granularity level of the explanation. It can be either one of `ALL_TOKENS` (includes all sentence tokens, even special tokens), `TOKEN`, `WORD`, or `SENTENCE`.
- `inference_mode` (`Callable[[torch.Tensor], torch.Tensor], optional`): The mode used for inference. It can be either one of `LOGITS`, `SOFTMAX`, or `LOG_SOFTMAX`. Use `InferenceModes` to choose the appropriate mode. Default `inference_mode` is `LOGITS`.
- `device` (`torch.device | None = None`): device on which the attribution method will be run,
- `args`: specifics for each method.

For gradient-based attribution methods, granularity is not calculated at the perturbation mask level, but at the aggregation level. We then aggregate the scores given on the tokens to create scores on the chosen granularity. There may be several ways of aggregating these scores, so we need a new argument, only for gradient-based methods:

- `granularity_aggregation_strategy` (`GranularityAggregationStrategy`): If granularity is set to 'WORD', this determines how to aggregate token-level attributions into word-level scores. Options are: 'MEAN', 'MAX', 'MIN', 'SUM' and 'SIGNED_MAX'.

**Step 2:** The `AttributionExplainer` class overloads the `__call__` method to directly invoke the `explain` function. Therefore, calling `explainer(inputs, targets)` is equivalent to `explainer.explain(inputs, targets)`. It takes two parameters:

- `inputs`: the samples for which explanations are requested.
- `targets`: specifies what to explain in the inputs. This can be a specific class or a set of classes (for classification tasks), or target texts (for generation tasks). If `targets=None`, the target is automatically inferred by performing a prediction on the input using the provided model.

## Available methods

➡️ **Inference-based Methods:**

- [Kernel SHAP](./methods/kernelshap.md): [Lundberg and Lee, 2017, A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874).
- [LIME](./methods/lime.md): [Ribeiro et al. 2013, "Why should i trust you?" explaining the predictions of any classifier](https://dl.acm.org/doi/abs/10.1145/2939672.2939778).
- [Occlusion](./methods/occlusion.md): [Zeiler and Fergus, 2014. Visualizing and understanding convolutional networks](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53).
- [Sobol Attribution](./methods/sobol.md): [Fel et al. 2021, Look at the variance! efficient black-box explanations with sobol-based sensitivity analysis](https://proceedings.neurips.cc/paper/2021/hash/da94cbeff56cfda50785df477941308b-Abstract.html).

🔁 **Gradient based methods:**

- [Integrated Gradient](./methods/integrated_gradients.md): [Sundararajan et al. 2017, Axiomatic Attribution for Deep Networks](http://proceedings.mlr.press/v70/sundararajan17a.html).
- [Saliency](./methods/saliency.md): [Simonyan et al. 2013, Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034).
- [SmoothGrad](./methods/smoothgrad.md): [Smilkov et al. 2017, SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)

# Custom API

You can easily use Interpreto with your attribution methods. You have to implement two things:

- a `Perturbator` to generate perturbed elements.
- an `Aggregator` to aggregate scores on perturbed elements.

Then, inherit from `MultitaskExplainerMixin` and `AttributionExplainer`, and finally, in the initialization of your method, point to your perturbator and aggregator in the `super().__init__`.

Please check how we did it with the implemented method. If you have any questions, please file an issue or contact us directly.

Once you succeed, please make a pull request. We welcome your method and contributions to the library.

```
perturbator = Perturbator(inputs_embedder)
aggregator = Aggregator()
explainer = AttributionExplainer(model, tokenizer, batch_size, perturbator, aggregator, device)
explanation = explainer(inputs, targets)
```

The API have two steps:

**Step 1:**
Instantiate a perturbator and an aggregator that will be used by the attribution method.

perturbator:

- `inputs_embedder`: Optional module used to embed input IDs when only ``input_ids`` are provided.

aggregator: no argument.

**Step 2:**
Instantiate the AttributionExplainer class with the perturbator and aggregator.

AttributionExplainer is the class that instantiates any attribution method based on whether or not it uses gradients (with the boolean argument `use_gradient`), its perturbation type and its aggregation type. It takes the custom parameters (`perturbator`, `aggregator`, `use_gradient`) and the classic parameters for explanation:

- `model` (`PreTrainedModel`): Hugging Face model to explain,
- `tokenizer` (`PreTrainedTokenizer`): Hugging Face tokenizer associated with the model,
- `batch_size` (`int`): batch size for the attribution method,
- `perturbator` (`BasePerturbator | None = None`): Instance used to generate input perturbations. If None, the perturbator returns only the original input,
- `aggregator` (`Aggregator | None = None`): Instance used to aggregate computed attribution scores. If None, the aggregator returns the original scores,
- `device` (`torch.device | None = None`): device on which the attribution method will be run,
- `use_gradient` (`bool`): If True, computes gradients instead of inference for targeted explanations.

The `AttributionExplainer` class overloads the `__call__` method to directly invoke explain. Thus, `explainer(inputs, targets)` is equivalent to `explainer.explain(inputs, targets)`. The explain method takes two parameters:

- `inputs`: the samples on which the explanations are requested, see inputs section for more detail.
- `targets`: another parameter to specify what to explain in the inputs, can be a specific class or a set of classes (for classification), or texts (for generation).
