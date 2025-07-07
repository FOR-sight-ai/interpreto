template: home.html
---

<img src="assets/img/interpreto_banner.png" alt="Interpreto: Interpretability Toolkit for LLMs" style="max-width:100%; height:auto;" />

[![Build status](https://img.shields.io/github/actions/workflow/status/FOR-sight-ai/interpreto/build.yml?branch=main)](https://github.com/FOR-sight-ai/interpreto/actions?query=workflow%3Abuild)
[![Docs status](https://img.shields.io/readthedocs/interpreto)](https://interpreto.readthedocs.io/)
[![Version](https://img.shields.io/pypi/v/interpreto?color=blue)](https://pypi.org/project/interpreto/)
[![Python Version](https://img.shields.io/pypi/pyversions/interpreto.svg?color=blue)](https://pypi.org/project/interpreto/)
[![Downloads](https://static.pepy.tech/badge/interpreto)](https://pepy.tech/project/interpreto)
[![License](https://img.shields.io/github/license/FOR-sight-ai/interpreto.svg)](https://github.com/FOR-sight-ai/interpreto/blob/main/LICENSE)
> **Warning**
>
> This library is currently in beta and many functions may not work. If you use it anyway, we welcome your comments; please open an issue!
>
> The API might change and the documentation is not up to date.
>
> In particular, it is not yet possible to obtain interpretable concept-based explanations.

## 📚 Table of contents

- [📚 Table of contents](#-table-of-contents)
- [🚀 Quick Start](#-quick-start)
- [📦 What's Included](#-whats-included)
- [👍 Contributing](#-contributing)
- [👀 See Also](#-see-also)
- [🙏 Acknowledgments](#-acknowledgments)
- [👨‍🎓 Creators](#-creators)
- [🗞️ Citation](#️-citation)
- [📝 License](#-license)

## 🚀 Quick Start

The library should be available on PyPI soon. Try `pip install interpreto` to install it.

Otherwise, you can clone the repository and install it locally with `pip install -e .`.

And any case, checkout the [attribution walkthrough](https://github.com/FOR-sight-ai/interpreto/tree/main/docs/notebooks/attribution_walkthrough.ipynb) and the [concept example](https://github.com/FOR-sight-ai/interpreto/tree/main/docs/notebooks/concept_examples.ipynb) to get started!

## 📦 What's Included

Interpreto 🪄 provides a modular framework encompassing Attribution Methods, Concept-Based Methods, and Evaluation Metrics.

### Attribution Methods

??? abstract "Interpreto includes both inference-based and gradient-based attribution methods:"

    **Inference-based Methods:**

    - Occlusion: [Zeiler and Fergus, 2014. Visualizing and understanding convolutional networks](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53).
    - LIME: [Ribeiro et al. 2013, "Why should i trust you?" explaining the predictions of any classifier](https://dl.acm.org/doi/abs/10.1145/2939672.2939778).
    - Kernel SHAP: [Lundberg and Lee, 2017, A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874).
    - Sobol Attribution: [Fel et al. 2021, Look at the variance! efficient black-box explanations with sobol-based sensitivity analysis](https://proceedings.neurips.cc/paper/2021/hash/da94cbeff56cfda50785df477941308b-Abstract.html).

    **Gradient based methods:**

    - Saliency: [Simonyan et al. 2013, Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034).
    - Integrated Gradient: [Sundararajan et al. 2017, Axiomatic Attribution for Deep Networks](http://proceedings.mlr.press/v70/sundararajan17a.html).
    - SmoothGrad: [Smilkov et al. 2017, SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)

    Will be implemented soon.

    - InputxGradient: [Simonyan et al. 2013, Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034).
    - DeepLift: [Shrikumar et al. 2017, Learning Important Features Through Propagating Activation Differences](http://proceedings.mlr.press/v70/shrikumar17a).
    - VarGrad: [Richter et al. 2020, VarGrad: A Low-Variance Gradient Estimator for Variational Inference](https://proceedings.neurips.cc/paper/2020/hash/9c22c0b51b3202246463e986c7e205df-Abstract.html)

### Concept-Based Methods

??? abstract "Concept-based explanations aim to provide high-level interpretations of latent model representations."

    Interpreto generalizes these methods through three core steps:

    1. Concept Discovery (e.g., from latent embeddings)

    2. Concept Interpretation (mapping discovered concepts to human-understandable elements)

    3. Concept-to-Output Attribution (assessing concept relevance to model outputs) [Work in progress]

    **Concept Discovery Techniques** (via [Overcomplete](https://github.com/KempnerInstitute/overcomplete)):

    - NMF, Semi-NMF, ConvexNMF

    - ICA, SVD, PCA

    - SAE variants (Vanilla SAE, TopK SAE, JumpReLU SAE, BatchTopK SAE)

    **Available Concept Interpretation Techniques:**

    - Top-k tokens from tokenizer vocabulary

    - Top-k tokens/words/sentences/samples from specific datasets

    *Concept Interpretation Techniques Added Soon:*

    - Input-to-concept attribution from dataset examples ([Jourdan et al. 2023](https://aclanthology.org/2023.findings-acl.317/))
    - Theme prediction via LLMs from top-k tokens/sentences

    *Concept Interpretation Techniques Added Later:*

    - OpenAI Interpretation ([Bills et al. 2023](https://openai.com/index/language-models-can-explain-neurons-in-language-models/))

    - Aligning concepts with human labels ([Sajjad et al. 2022](https://aclanthology.org/2022.naacl-main.225/))

    - Word cloud visualizations of concepts ([Dalvi et al. 2022](https://arxiv.org/abs/2205.07237))

    - VocabProj & TokenChange ([Gur-Arieh et al. 2025](https://arxiv.org/abs/2501.08319))

    **Concept-to-Output Attribution:**

    This part will be implemented later, but all the attribution methods presented above will be available here.

    *Note that only methods with a concept extraction that has an encoder (input to concept) AND a decoder (concept to output) can use this function.*

    **Specific methods:**

    **[Available later when all parts are implemented]** Thanks to this generalization encompassing all concept-based methods and our highly flexible architecture, we can easily obtain a large number of concept-based methods:

    - CAV and TCAV: [Kim et al. 2018, Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)](http://proceedings.mlr.press/v80/kim18d.html)

    - ConceptSHAP: [Yeh et al. 2020, On Completeness-aware Concept-Based Explanations in Deep Neural Networks](https://proceedings.neurips.cc/paper/2020/hash/ecb287ff763c169694f682af52c1f309-Abstract.html)

    - COCKATIEL: [Jourdan et al. 2023, COCKATIEL: COntinuous Concept ranKed ATtribution with Interpretable ELements for explaining neural net classifiers on NLP](https://aclanthology.org/2023.findings-acl.317/)

    - Yun et al. 2021, [Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors](https://arxiv.org/abs/2103.15949)

    - FFN values interpretation: [Geva et al. 2022, Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space](https://aclanthology.org/2022.emnlp-main.3/)

    - SparseCoding: [Cunningham et al. 2023, Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600)

    - Parameter Interpretation: [Dar et al. 2023, Analyzing Transformers in Embedding Space](https://aclanthology.org/2023.acl-long.893/)

### Evaluation Metrics

**Evaluation Metrics for Attribution**

We don't yet have metrics implemented for attribution methods, but that's coming soon!

**Evaluation Metrics for Concepts**

??? abstract "Several properties of the concept-space are desirable."

    List of properties and corresponding metrics:

    - *Concept-space faithfulness:* `MSE`, `FID`, or define a custom one through `ReconstructionError` by specifying a `reconstruction_space` and a `distance_function`.

    - *Concept-space complexity:* `Sparsity` and `SparsityRatio` metric are available.

    - *Concept-space stability:* You can use `Stability` metric to compare concept-model dictionaries.

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

Interpreto 🪄 is a project of the FOR and the [DEEL](https://www.deel.ai) teams at the [IRT Saint-Exupéry](https://www.irt-saintexupery.com/) in Toulouse, France.

## 🗞️ Citation

If you use Interpreto 🪄 as part of your workflow in a scientific publication, please consider citing 🗞️ our paper (coming soon):

```bibtex
BibTeX entry coming soon
```

## 📝 License

The package is released under [MIT license](LICENSE).
