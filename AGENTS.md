# AGENTS.md

## Goal

`interpreto` is a modular interpretability toolkit for transformer models. The repository aims to provide:

- an easy-to-use public API for attribution and concept-based explanations,
- detailed documentation with concrete examples,
- precise internal representations for tensors, targets, and activations,
- reusable building blocks that can be combined without rewriting the whole pipeline.

The main product surface is:

- attribution methods for classification and generation,
- concept discovery and concept interpretation workflows,
- evaluation metrics,
- HTML visualizations,
- docs and notebooks showing real usage.

## Repository Map

- `interpreto/__init__.py`
  - Curated public API. If a feature is meant to be user-facing, it usually belongs here too.
- `interpreto/concepts.splitters/`
  - Bridges raw Hugging Face models to Interpreto internals.
  - `inference_wrapper.py`: shared batching, device handling, logits/gradient access, padding helpers.
  - `classification_inference_wrapper.py`: targeted scoring for classification tasks.
  - `generation_inference_wrapper.py`: targeted scoring for generation tasks.
  - `model_with_split_points.py`: `nnsight`-based model splitting and activation extraction for concept methods.
  - `llm_interface.py`: abstraction layer for LLM-based concept labeling.
- `interpreto/attributions/`
  - Attribution framework.
  - `base.py`: shared explainers, normalization, output dataclasses, classification/generation glue.
  - `methods/`: LIME, KernelShap, Occlusion, Sobol, Saliency, Integrated Gradients, SmoothGrad, etc.
  - `perturbations/`: perturbation generators used by attribution methods.
  - `aggregations/`: score aggregation logic.
  - `metrics/`: insertion/deletion evaluation.
- `interpreto/concepts/`
  - Concept-based interpretability framework.
  - `base.py`: base concept explainer interfaces.
  - `methods/`: neurons-as-concepts, overcomplete/SAE methods, sklearn-based methods, Cockatiel.
  - `interpretations/`: `TopKInputs`, `LLMLabels`, and related interpretation utilities.
  - `metrics/`: reconstruction, sparsity, stability, and ConSim.
- `interpreto/commons/`
  - Shared utilities such as granularity handling, generator helpers, and distances.
- `interpreto/typing.py`
  - Central typing aliases and protocols. This file expresses the intended normalized internal shapes and interfaces.
- `interpreto/visualizations/`
  - HTML/CSS/JS renderers for attribution and concept outputs.
  - Visualizations should consume normalized outputs, not recompute model logic.
- `interpreto/_vendor/overcomplete/`
  - Vendored dependency for concept learning backends. Avoid touching it unless the change really belongs there.
- `tests/`
  - Pytest suite. Reuse fixtures from `tests/conftest.py` whenever possible.
- `docs/`
  - MkDocs source, API pages, and notebooks.
- `site/`
  - Generated documentation output. Prefer editing `docs/`, not `site/`.

## Key Dependencies

- `torch`
  - Core tensor and model execution backend.
- `transformers`
  - Main model/tokenizer interface and public compatibility target.
- `nnsight`
  - Used by `ModelWithSplitPoints` for split points and activation capture.
- `jaxtyping` and `beartype`
  - Preferred tools for explicit tensor typing and shape contracts.
- `scikit-learn`, `scipy`, `einops`, `matplotlib`, `nltk`
  - Supporting libraries for methods, metrics, preprocessing, and visualization.
- `bitsandbytes`
  - Compatibility with quantized transformer loading.
- `mkdocs` stack
  - Documentation build system.

## How The Pieces Interact

### Attribution pipeline

User inputs can arrive in several formats: strings, tokenized mappings, tensors, or iterables of those. The code should normalize them early, then keep core computations on one internal format.

Typical flow:

1. User input and targets enter an attribution explainer from `interpreto.attributions`.
2. The explainer normalizes inputs/targets in `attributions/base.py`.
3. A perturbator or gradient path generates the computation stream.
4. A task-specific inference wrapper computes targeted logits or gradients.
5. An aggregator converts raw scores into final attribution values.
6. The result is packaged as `AttributionOutput`.
7. Metrics and visualizations consume `AttributionOutput`.

Important style point: attribution code is intentionally generator-friendly. Many paths are designed to work sample by sample or batch by batch instead of materializing everything eagerly. Preserve that when making changes, especially for generation and prompt construction logic.

### Concept pipeline

Typical flow:

1. `ModelWithSplitPoints` wraps a transformer model and exposes split points.
2. `get_activations()` extracts latent activations at a chosen granularity.
3. A concept explainer from `interpreto.concepts.methods` fits or applies a concept model on those activations.
4. Interpretation methods such as `TopKInputs` or `LLMLabels` map concept dimensions to human-readable descriptions.
5. Metrics and visualizations operate on the resulting concept-space artifacts.

`ModelWithSplitPoints` is the bridge between the transformer world and concept methods. Most concept changes should respect that layering instead of bypassing it.

### Granularity and normalization

Granularity is a core abstraction shared across attribution and concept code. The code often accepts flexible user inputs, but should converge quickly toward:

- normalized `TensorMapping`-style model inputs,
- normalized target tensors,
- normalized activation tensors,
- normalized output dataclasses.

This repository prefers a flexible public API and a stricter internal core.

## Repository Vibe

- Keep the public API easy to use.
  - Users may provide several input formats.
  - Internal computations should still be normalized into a single clear format as early as possible.
- Prefer precise typing.
  - `jaxtyping` is valuable here because tensor shapes matter a lot for readability and debugging.
  - Be pragmatic at boundaries with `transformers` and `nnsight`; do not make the code worse just to force shape annotations through awkward external APIs.
- Documentation matters.
  - Detailed docstrings, examples, file-level comments, and inline comments are a feature of the repository, not noise.
  - When adding or changing logic, explain the shape conventions and the intent, especially around generators, token alignment, split points, and concept encoding.
- The repository is modular.
  - Prefer plug-and-play building blocks over special-purpose monoliths.
  - Reuse wrappers, perturbators, aggregators, metrics, and visualization outputs rather than duplicating logic.
- Prefer one place for validation.
  - Do not add repeated guardrails in every layer if the check already belongs at the public boundary or is already enforced by typing/contracts.
  - Re-check only if a lower-level function can be called independently or if the invariant genuinely changes.
- Smaller changes are usually better.
  - Do not refactor by default.
  - If a minimal patch would conflict with the method/class/repository design, then do the slightly larger coherent refactor instead of adding a local hack.
- Keep implementations efficient but simple.
  - Prefer straightforward Torch code.
  - If a much faster version would add a lot of complexity, it is often better to land the clean version first and leave a focused `TODO`.
- In attribution code, preserve the generator-based pipeline mindset.
  - The repository often processes attribution sample by sample, while trying to construct good prompts and avoid unnecessary materialization.

## Coding Expectations

- Write docstrings and the important inline comments at the same time as the code change, or before.
- Prefer file-level comments when the whole module has a specific role or subtle invariant.
- Keep internal data formats explicit.
- If adding a new public class or function, check whether it should be re-exported in a package `__init__.py` and documented in `docs/`.
- Use the existing module boundaries.
  - New attribution methods usually belong in `interpreto/attributions/methods/`.
  - New perturbation logic belongs in `interpreto/attributions/perturbations/`.
  - New concept methods belong in `interpreto/concepts/methods/`.
  - New interpretation strategies should use the existing concept explainer interfaces.

## Tests

Testing style in this repository is usually a mix of:

- method-level tests for specific algorithmic behavior,
- class-level tests for API and integration behavior,
- sanity checks for end-to-end invariants.

Guidelines:

- For a new feature, test-driven development is preferred when practical.
- Keep tests reviewable. Do not add large numbers of nearly identical tests.
- Be very clear in test comments/docstrings about what the test is proving.
- Reuse `tests/conftest.py`, `tests/fixtures/`, and existing helpers before inventing new scaffolding.
- Prefer `hf-internal-testing/*` tiny models over large custom placeholders or long fake model definitions.
- Do not test the same invariant in many places unless it protects distinct call paths.

## Change Workflow For Agents

1. Think first.
   - Understand which layer should change.
   - Prefer the smallest coherent modification.
   - If the design tradeoff is uncertain, it is better to ask for an opinion than to guess.
2. Add or update tests.
   - For new features or bug fixes, start from the behavior you want to lock in.
   - Reuse fixtures and tiny test models whenever possible.
3. Implement the change.
   - Keep the code aligned with existing abstractions.
   - Avoid clever one-off tricks that only satisfy the immediate patch.
4. Update documentation if needed.
   - Public API changes usually need docstring and docs updates.
   - Example-driven documentation is part of the repository style.
5. Verify with targeted commands first.

Useful commands:

- `make install-dev`
- `make lint`
- `make fast-test`
- `make test-cpu`
- `python -m pytest -n auto -c pyproject.toml -v path/to/test_file.py`

## Practical Do / Don't

Do:

- Normalize flexible user inputs into one internal format early.
- Use `jaxtyping` where it improves shape clarity.
- Preserve generator-based or streaming-friendly flows.
- Add comments where tensor shapes, batching, or prompt construction are non-obvious.
- Favor small coherent patches.

Don't:

- Add redundant guardrails in every layer.
- Materialize huge intermediate lists if the existing pipeline is intentionally iterable/generator-based.
- Refactor broadly without a concrete design reason.
- Fight external library APIs just to satisfy an idealized typing style.
- Edit generated docs in `site/` when the real source lives in `docs/`.
