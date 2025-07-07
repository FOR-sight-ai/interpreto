# 📍 Interpreto Roadmap

Welcome to the roadmap of Interpreto 🪄. This document outlines the planned features and improvements for upcoming releases.

---

## 🧭 Upcoming Features

### 1. Evaluation Metrics for Attribution Methods

We plan to integrate a set of standardized evaluation metrics to assess the quality and reliability of attribution methods. We'll start by adding the **insertion/deletion metrics**.

### 2. Gradient-based Attribution: Token and Word Granularity

We aim to enhance our **gradient-based attribution methods** (e.g., Integrated Gradients, Saliency) by adding:

- **Token-level attribution**
- **Word-level attribution**

We currently have these attribution methods only with the ALL_TOKENS granularity, which corresponds to having an attribution score for each token in the sentence (even special tokens). We want to align gradient-based methods with inference-based ones.

### 3. Integration of VarGrad

We will implement **VarGrad**, a variance-based extension of gradient attribution. This method improves robustness by computing the **variance of attributions across multiple noisy forward passes**, capturing uncertainty and stabilizing explanations.

### 4. LLM-Based Concept Interpretation: Theme Prediction

We will introduce a novel interpretation method for concept vectors using Large Language Models (LLMs):

- The **"theme prediction"** method will generate human-readable summaries of concept vectors by prompting an LLM with Top-K examples (words, tokens, or sentences)
- This offers a **natural language explanation** of abstract concepts, improving transparency and user comprehension

---

## 🙌 Contribute

Want to help shape the future of Interpreto? Check out our [contributing guide](contributing.md) and feel free to open an issue or pull request!
