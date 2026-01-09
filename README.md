<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/img/interpreto_banner_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/img/interpreto_banner.png">
  <img src="docs/assets/img/interpreto_banner_dark.png" alt="Interpreto: Interpretability Toolkit for LLMs" />
</picture>

[![Build status](https://img.shields.io/github/actions/workflow/status/FOR-sight-ai/interpreto/build.yml?branch=main)](https://github.com/FOR-sight-ai/interpreto/actions?query=workflow%3Abuild)
[![Version](https://img.shields.io/pypi/v/interpreto?color=blue)](https://pypi.org/project/interpreto/)
[![Python Version](https://img.shields.io/pypi/pyversions/interpreto.svg?color=blue)](https://pypi.org/project/interpreto/)
[![Downloads](https://static.pepy.tech/badge/interpreto)](https://pepy.tech/project/interpreto)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/FOR-sight-ai/interpreto/blob/main/LICENSE)

  <!-- Link to the documentation -->
  <a href="https://for-sight-ai.github.io/interpreto/"><strong>Explore Interpreto docs »</strong></a>
  <br>

</div>

## 🚀 Quick Start

The library is available on PyPI, try `pip install interpreto` to install it.

Checkout the tutorials to get started:

- [Attributions walkthrough](https://github.com/FOR-sight-ai/interpreto/tree/main/docs/notebooks/attribution_walkthrough.ipynb) (both classification and generation)
- [Classification concept-based explanations](https://github.com/FOR-sight-ai/interpreto/tree/main/docs/notebooks/classification_concepts_tutorial.ipynb)
- [Generation concept-based explanations](https://github.com/FOR-sight-ai/interpreto/tree/main/docs/notebooks/generation_concepts_tutorial.ipynb)

## 📦 What's Included

Interpreto 🪄 provides a modular framework encompassing Attribution Methods, Concept-Based Methods, and Evaluation Metrics.

### Attribution Methods

Interpreto includes both inference-based and gradient-based attribution methods.

They all work seamlessly for both classification (`...ForSequenceClassification`) and generation (`...ForCausalLM`)

**Inference-based Methods:**

- [`KernelShap`](https://for-sight-ai.github.io/interpreto/api/attributions/methods/kernel_shap/) — [Lundberg and Lee, 2017](https://arxiv.org/abs/1705.07874)
- [`LIME`](https://for-sight-ai.github.io/interpreto/api/attributions/methods/lime/) — [Ribeiro et al., 2013](https://dl.acm.org/doi/abs/10.1145/2939672.2939778)
- [`Occlusion`](https://for-sight-ai.github.io/interpreto/api/attributions/methods/occlusion/) — [Zeiler and Fergus, 2014](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53)
- [`Sobol`](https://for-sight-ai.github.io/interpreto/api/attributions/methods/sobol/) — [Fel et al., 2021](https://proceedings.neurips.cc/paper/2021/hash/da94cbeff56cfda50785df477941308b-Abstract.html)

**Gradient-based methods:**

- [`GradientShap`](https://for-sight-ai.github.io/interpreto/api/attributions/methods/gradient_shap/) — [Lundberg and Lee, 2017](https://arxiv.org/abs/1705.07874)
- [`InputxGradient`](https://for-sight-ai.github.io/interpreto/api/attributions/methods/inputx_gradient/) — [Simonyan et al., 2013](https://arxiv.org/abs/1312.6034)
- [`Integrated Gradient`](https://for-sight-ai.github.io/interpreto/api/attributions/methods/integrated_gradient/) — [Sundararajan et al., 2017](http://proceedings.mlr.press/v70/sundararajan17a.html)
- [`Saliency`](https://for-sight-ai.github.io/interpreto/api/attributions/methods/saliency/) — [Simonyan et al., 2013](https://arxiv.org/abs/1312.6034)
- [`SmoothGrad`](https://for-sight-ai.github.io/interpreto/api/attributions/methods/smooth_grad/) — [Smilkov et al., 2017](https://arxiv.org/abs/1706.03825)
- [`SquareGrad`](https://for-sight-ai.github.io/interpreto/api/attributions/methods/square_grad/) — [Hooker et al., 2019](https://arxiv.org/abs/1806.10758)
- [`VarGrad`](https://for-sight-ai.github.io/interpreto/api/attributions/methods/var_grad/) — [Richter et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/9c22c0b51b3202246463e986c7e205df-Abstract.html)

### Concept-Based Methods or Mechanistic Interpretability

Concept-based explanations aim to provide high-level interpretations of latent model representations.

Interpreto generalizes these methods through three core steps:

1. Concept Discovery (e.g., from latent embeddings)
2. Concept Interpretation (mapping discovered concepts to human-understandable elements)
3. Concept-to-Output Attribution (assessing concept relevance to model outputs)

**Dictionary Learning for Concept Discovery** (mainly via [Overcomplete](https://github.com/KempnerInstitute/overcomplete)):

- Interpret neurons directly via [`NeuronsAsConcepts`](https://for-sight-ai.github.io/interpreto/api/concepts/methods/neurons_as_concepts/)
- [`NMF`](https://for-sight-ai.github.io/interpreto/api/concepts/methods/optim/#interpreto.concepts.NMFConcepts), [`Semi-NMF`](https://for-sight-ai.github.io/interpreto/api/concepts/methods/optim/#interpreto.concepts.SemiNMFConcepts), [`ConvexNMF`](https://for-sight-ai.github.io/interpreto/api/concepts/methods/optim/#interpreto.concepts.ConvexNMFConcepts)
- [`ICA`](https://for-sight-ai.github.io/interpreto/api/concepts/methods/optim/#interpreto.concepts.ICAConcepts), [`SVD`](https://for-sight-ai.github.io/interpreto/api/concepts/methods/optim/#interpreto.concepts.SVDConcepts), [`PCA`](https://for-sight-ai.github.io/interpreto/api/concepts/methods/optim/#interpreto.concepts.PCAConcepts), [`KMeans`](https://for-sight-ai.github.io/interpreto/api/concepts/methods/optim/#interpreto.concepts.KMeansConcepts)
- SAE variants: [`Vanilla SAE`](https://for-sight-ai.github.io/interpreto/api/concepts/methods/sae/#interpreto.concepts.VanillaSAEConcepts), [`TopK SAE`](https://for-sight-ai.github.io/interpreto/api/concepts/methods/sae/#interpreto.concepts.TopKSAEConcepts), [`JumpReLU SAE`](https://for-sight-ai.github.io/interpreto/api/concepts/methods/sae/#interpreto.concepts.JumpReLUSAEConcepts), [`BatchTopK SAE`](https://for-sight-ai.github.io/interpreto/api/concepts/methods/sae/#interpreto.concepts.BatchTopKSAEConcepts)

**Available Concept Interpretation Techniques:**

- Top-k tokens from tokenizer vocabulary via [`TopKInputs`](https://for-sight-ai.github.io/interpreto/api/concepts/concepts_interpretations/#interpreto.concepts.interpretations.TopKInputs) and `use_vocab=True`
- Top-k tokens/words/sentences/samples from specific datasets via [`TopKInputs`](https://for-sight-ai.github.io/interpreto/api/concepts/concepts_interpretations/#interpreto.concepts.interpretations.TopKInputs)
- Label concepts via LLMs with [`LLMLabels`](https://for-sight-ai.github.io/interpreto/api/concepts/concepts_interpretations/#interpreto.concepts.interpretations.LLMLabels) ([Bills et al. 2023](https://openai.com/index/language-models-can-explain-neurons-in-language-models/))

<details><summary>Concept Interpretation Techniques Added in the future:</summary>

- Input-to-concept attribution from dataset examples ([Jourdan et al. 2023](https://aclanthology.org/2023.findings-acl.317/))
- Theme prediction via LLMs from top-k tokens/sentences
- Aligning concepts with human labels ([Sajjad et al. 2022](https://aclanthology.org/2022.naacl-main.225/))
- Word cloud visualizations of concepts ([Dalvi et al. 2022](https://arxiv.org/abs/2205.07237))
- VocabProj & TokenChange ([Gur-Arieh et al. 2025](https://arxiv.org/abs/2501.08319))

</details>

**Concept-to-Output Attribution:**

Estimate the contribution of each concept to the model output.

Can be obtained with any concept-based explainer via [`MethodConcepts.concept_output_gradient()`](https://for-sight-ai.github.io/interpreto/api/concepts/methods/base/#interpreto.concepts.ConceptAutoEncoderExplainer.concept_output_gradient).

<details><summary><b>Papers available in the future:</b></summary>

Thanks to this generalization encompassing all concept-based methods and our highly flexible architecture, we can easily obtain a large number of concept-based methods:

- CAV and TCAV: [Kim et al. 2018, Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)](http://proceedings.mlr.press/v80/kim18d.html)
- ConceptSHAP: [Yeh et al. 2020, On Completeness-aware Concept-Based Explanations in Deep Neural Networks](https://proceedings.neurips.cc/paper/2020/hash/ecb287ff763c169694f682af52c1f309-Abstract.html)
- COCKATIEL: [Jourdan et al. 2023, COCKATIEL: COntinuous Concept ranKed ATtribution with Interpretable ELements for explaining neural net classifiers on NLP](https://aclanthology.org/2023.findings-acl.317/)
- Yun et al. 2021, [Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors](https://arxiv.org/abs/2103.15949)
- FFN values interpretation: [Geva et al. 2022, Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space](https://aclanthology.org/2022.emnlp-main.3/)
- SparseCoding: [Cunningham et al. 2023, Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600)
- Parameter Interpretation: [Dar et al. 2023, Analyzing Transformers in Embedding Space](https://aclanthology.org/2023.acl-long.893/)

</details>

### Evaluation Metrics

**Evaluation Metrics for Attribution**

To evaluate attribution methods faithfulness, there are the [`Insertion`](https://for-sight-ai.github.io/interpreto/api/attributions/metrics/insertion/) and [`Deletion`](https://for-sight-ai.github.io/interpreto/api/attributions/metrics/deletion/) metrics.

**Evaluation Metrics for Concepts**

Concept-based methods have several steps that can be evaluated together via [`ConSim`](https://for-sight-ai.github.io/interpreto/api/concepts/metrics/consim/).

Or independently:

- Concept-space (dictionary learning evaluation)
  - faithfulness: [`MSE`](https://for-sight-ai.github.io/interpreto/api/concepts/metrics/reconstruction_metrics/#interpreto.concepts.metrics.MSE), [`FID`](https://for-sight-ai.github.io/interpreto/api/concepts/metrics/reconstruction_metrics/#interpreto.concepts.metrics.FID), and [`ReconstructionError`](https://for-sight-ai.github.io/interpreto/api/concepts/metrics/reconstruction_metrics/#interpreto.concepts.metrics.ReconstructionError)
  - complexity: [`Sparsity`](https://for-sight-ai.github.io/interpreto/api/concepts/metrics/sparsity_metrics/#interpreto.concepts.metrics.Sparsity), [`SparsityRatio`](https://for-sight-ai.github.io/interpreto/api/concepts/metrics/sparsity/), [`SparsityRatio`](https://for-sight-ai.github.io/interpreto/api/concepts/metrics/sparsity/#interpreto.concepts.metrics.SparsityRatio)
  - stability: [`Stability`](https://for-sight-ai.github.io/interpreto/api/concepts/metrics/dictionary_metrics/#interpreto.concepts.metrics.Stability)
- Concepts interpretations
  - No metric yet, will be included soon.
- Concept-to-Output attribution
  - No metric yet, will be included soon.

## 👍 Contributing

Feel free to propose your ideas or come and contribute with us on the Interpreto 🪄 toolbox! We have a specific document where we describe in a simple way how to make your [first pull request](docs/contributing.md).

## 👀 See Also

More from the DEEL project:

- [Xplique](https://github.com/deel-ai/xplique) a Python library dedicated to explaining neural networks (Images, Time Series, Tabular data) on TensorFlow.
- [Puncc](https://github.com/deel-ai/puncc) a Python library for predictive uncertainty quantification using conformal prediction.
- [oodeel](https://github.com/deel-ai/oodeel) a Python library that performs post-hoc deep Out-of-Distribution (OOD) detection on already trained neural network image classifiers.
- [deel-lip](https://github.com/deel-ai/deel-lip) a Python library for training k-Lipschitz neural networks on TensorFlow.
- [deel-torchlip](https://github.com/deel-ai/deel-torchlip) a Python library for training k-Lipschitz neural networks on PyTorch.
- [Influenciae](https://github.com/deel-ai/influenciae) a Python library dedicated to computing influence values for the discovery of potentially problematic samples in a dataset.
- [DEEL White paper](https://arxiv.org/abs/2103.10529) a summary of the DEEL team on the challenges of certifiable AI and the role of data quality, representativity and explainability for this purpose.

## 🙏 Acknowledgments

This project received funding from the French ”Investing for the Future – PIA3” program within the Artificial and Natural Intelligence Toulouse Institute (ANITI). The authors gratefully acknowledge the support of the [DEEL](https://www.deel.ai) and the FOR projects.

## 👨‍🎓 Creators

Interpreto 🪄 is a project of the [FOR](https://www.irt-saintexupery.com/fr/for-program/) and the [DEEL](https://www.deel.ai) teams at the [IRT Saint-Exupéry](https://www.irt-saintexupery.com/) in Toulouse, France.

## 🗞️ Citation

If you use Interpreto 🪄 as part of your workflow in a scientific publication, please consider citing 🗞️ our paper:

```bibtex
@article{poche2025interpreto,
    title       = {Interpreto: An Explainability Library for Transformers},
    author      = {Poch{\'e}, Antonin and Mullor, Thomas and Sarti, Gabriele and Boisnard, Fr{\'e}d{\'e}ric and Friedrich, Corentin and Claye, Charlotte and Hoofd, Fran{\c{c}}ois and Bernas, Raphael and Hudelot, C{\'e}line and Jourdan, Fanny},
    journal     = {arXiv preprint arXiv:2512.09730},
    year        = {2025}
}
```

## 📝 License

The package is released under [MIT license](LICENSE).
