---
icon: material/text-short
---

# Concept Explanation Metrics

As described in the [Concept Explainers](../overview.md) page, concept-based explanations are obtain by applying several steps:

- Defining the concept-space, usually through dictionary learning.

- Interpreting the direction of the concept space.

- Eventually, measuring the importance of each concept in the prediction.

Concept-based metrics evaluate either one of the two first components, or both.
To evaluate the importance of a concept, attribution metrics are used.

Each metric takes in different arguments, hence no common API can be defined, apart from the `compute` method:

```python
from interpreto.concepts.metrics import MetricClass

metric = MetricClass(...)
score = metric.compute(...)
```

<table>
  <thead>
    <tr>
      <th>Metric family</th>
      <th>Property</th>
      <th>Metrics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="../dictionary_metrics/">Dictionary Metrics</a></td>
      <td>Concept-space Stability</td>
      <td>
        <a href="../dictionary_metrics/#stability">Stability</a>
      </td>
    </tr>
    <tr>
      <td><a href="../reconstruction_metrics/">Reconstruction Metrics</a></td>
      <td>Concept-space Faithfulness</td>
      <td>
        <a href="../reconstruction_metrics/#mse">MSE</a><br>
        <a href="../reconstruction_metrics/#fid">FID</a><br>
        <a href="../reconstruction_metrics/#custom">Custom</a>
      </td>
    </tr>
    <tr>
      <td><a href="../sparsity_metrics/">Sparsity Metrics</a></td>
      <td>Concept-space Complexity</td>
      <td>
        <a href="../sparsity_metrics/#sparsity">Sparsity</a><br>
        <a href="../sparsity_metrics/#sparsity-ratio">Sparsity Ratio</a>
      </td>
    </tr>
  </tbody>
</table>

## Evaluating the concept-space

The concept space is define by a concept model encoding latent activations into concept activations.
Some concept models can also reconstruct the latent activations from the concept activations.

Several properties of the concept-space are desirable:

- The concept-space should be faithful to the latent space data distribution.

- The concept-space should have a low complexity to push toward interpretability.

- The concept-space is stable across different training regimes.

### Concept-space faithfulness

Concept-space faithfulness is often measured via the reconstruction error of the latent activations when projected back and forth in the concept space.
Either in the latent space or in the logits space. The distance used to compare activations has a big impact on what is measured.

In `interpreto` you can use the [ReconstructionError](reconstruction_metrics.md#custom) to define a custom metric by specifying a `reconstruction_space` and a `distance_function`.
Or you can use the [MSE](reconstruction_metrics.md#mse) or [FID](reconstruction_metrics.md#fid) metrics.

### Concept-space complexity

The concept-space complexity is often measured via the sparsity of its activations.

In `interpreto` you can use: [Sparsity](sparsity_metrics.md#sparsity) and [SparsityRatio](sparsity_metrics.md#sparsity-ratio).

### Concept-space stability

The concept-space should stay the same across different training regimes.
*i.e.* with different seeds, different data splits...

In `interpreto` you can use: [Stability](dictionary_metrics.md#stability) to compare concept-model dictionaries.
