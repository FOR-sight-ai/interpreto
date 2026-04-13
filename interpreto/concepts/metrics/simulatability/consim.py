# MIT License
#
# Copyright (c) 2025 IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL and FOR are research programs operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import torch

from interpreto.concepts.metrics.simulatability.base import AutomatedSimulatability


class PromptSetting(NamedTuple):
    """
    Low-level configuration of a ConSim prompt.

    Each flag enables one prompt block. `PromptTypes` exposes the common presets used in papers and
    tests, while direct `PromptSetting(...)` instances let advanced users define custom ablations.

    Attributes:
        concepts_global_importances: bool
            Include per-class concept summaries.
        global_contrastive_importances: bool
            Include fact-versus-foil global concept summaries instead of per-class summaries.
            Incompatible with `concepts_global_importances`.
        lp_samples: bool
            Include learning-phase examples in the shared system prompt.
        lp_concepts_local_contributions: bool
            Add local concept contributions for each learning-phase example.
        lp_local_contrastive_importance: bool
            Add contrastive local contributions for each learning-phase example.
            Contrastive are shown for errors and classic contributions for correct predictions.
            Incompatible with `lp_concepts_local_contributions`.
        anonymize_classes: bool
            Replace user-facing class names with `Class_i`.
            Preventing the LLM from using knowledge on classes names.
    """

    # initial phase
    concepts_global_importances: bool = False
    global_contrastive_importances: bool = False

    # learning phase
    lp_samples: bool = False
    lp_concepts_local_contributions: bool = False
    lp_local_contrastive_importance: bool = False

    # anonymization
    anonymize_classes: bool = False

    def validate(
        self,
        *,
        concepts_interpretation: dict[int, str],
        labels: torch.Tensor | list[int] | None,
    ) -> None:
        """
        Validate internal consistency for a prompt setting.

        This method only checks setting-level constraints, such as mutually exclusive options or
        inputs required by a given prompt family. Tensor shape checks are handled separately in
        `ConSim._check_input_settings_correspondence` so callers fail before any prompt text is
        rendered.

        Arguments:
            concepts_interpretation: dict[int, str]
                Reserved for API symmetry with prompt builders. It is not used yet.
            labels: torch.Tensor | list[int] | None
                Gold labels aligned with the selected samples. Required for contrastive local
                prompts.

        Raises:
            ValueError:
                If the setting is inconsistent or requires missing inputs.
        """
        _ = concepts_interpretation
        if self.concepts_global_importances and self.global_contrastive_importances:
            raise ValueError(
                "PromptSetting.concepts_global_importances and PromptSetting.global_contrastive_importances are mutually exclusive."
            )

        if self.lp_concepts_local_contributions and self.lp_local_contrastive_importance:
            raise ValueError(
                "PromptSetting.lp_concepts_local_contributions and PromptSetting.lp_local_contrastive_importance are mutually exclusive."
            )

        if self.lp_local_contrastive_importance and labels is None:
            raise ValueError(
                "PromptSetting.lp_local_contrastive_importance=True requires `labels` to be provided to ConSim.construct_prompt()."
            )

        if self.lp_concepts_local_contributions or self.lp_local_contrastive_importance:
            if not self.lp_samples:
                raise ValueError(
                    "PromptSetting.lp_concepts_local_contributions or PromptSetting.lp_local_contrastive_importance "
                    "requires `lp_samples=True` to be provided to ConSim.construct_prompt()."
                )


class PromptTypes(Enum):
    """
    Named ConSim prompt presets.

    Naming convention:
        - `L*`: baselines without concept explanations.
        - `E*`: standard concept explanations.
        - `C*`: contrastive concept explanations.
        - `with_lp` / `without_lp`: whether learning-phase examples are included.

    Each enum value is a `PromptSetting`. Use the enum for standard experiments and direct
    `PromptSetting(...)` values for custom studies.
    """

    L1_baseline_without_lp = PromptSetting()
    E1_global_concepts_without_lp = PromptSetting(concepts_global_importances=True)
    L2_baseline_with_lp = PromptSetting(lp_samples=True)
    E2_global_concepts_with_lp = PromptSetting(concepts_global_importances=True, lp_samples=True)
    E3_global_and_local_concepts_with_lp = PromptSetting(
        concepts_global_importances=True,
        lp_samples=True,
        lp_concepts_local_contributions=True,
    )
    C1_contrastive_global_concepts_without_lp = PromptSetting(
        global_contrastive_importances=True,
    )
    C2_contrastive_global_concepts_with_lp = PromptSetting(
        global_contrastive_importances=True,
        lp_samples=True,
    )
    C3_contrastive_global_and_local_concepts_with_lp = PromptSetting(
        global_contrastive_importances=True,
        lp_samples=True,
        lp_local_contrastive_importance=True,
    )
    C4_contrastive_local_concepts = PromptSetting(
        concepts_global_importances=True,
        lp_samples=True,
        lp_local_contrastive_importance=True,
    )
    C5_contrastive_local_only = PromptSetting(
        lp_samples=True,
        lp_local_contrastive_importance=True,
    )


class ConSim(AutomatedSimulatability):
    """
    ConSim prompt builder for concept-based automated simulatability.

    ConSim measures whether concept explanations help a meta-predictor reproduce a classifier's
    outputs. In this module, `ConSim` is responsible only for ConSim-specific prompt
    design and validation. It does not compute model predictions, concept importances, or call the
    LLM on its own.

    Therefore, users need to compute model predictions and concept explanations beforehand.

    Typical workflow:
        1. Instantiate `ConSim(classes=...)`.
        2. Call `select_examples(...)` on precomputed inputs, labels, and model predictions.
        3. Use a fitted concept explainer upstream to build the explanation artifacts required by
           the chosen setting, for example with
           `TopKInputs(concept_explainer).interpret(...)` and
           `concept_explainer.concept_output_gradient(...)`:
           `concepts_interpretation`, `global_importances`, optional `local_importances`, and
           optional `contrastive_pairs`.
        4. Call `construct_prompt(...)`.
        5. Run the prompts through your LLM interface outside this class.
        6. Compute responses with `llm_interface.batch_generate(...)`.
        7. Score the returned responses with `score_from_responses(...)`.

    Reference:
        A. Poché, A. Jacovi, A.M. Picard, V. Boutin, and F. Jourdan. ConSim: Measuring
        Concept-Based Explanations' Effectiveness with Automated Simulatability. ACL 2025.
        https://aclanthology.org/2025.acl-long.279/

    Arguments:
        classes: list[str]
            Display names for class ids. Inherited from `AutomatedSimulatability`; `classes[i]`
            must match class id `i`.

    Attributes:
        classes: list[str]
            Display names for class ids.
        prompt_types: type[PromptTypes]
            Preset prompt configurations shipped with ConSim.
            These are prompt settings that can be passed to `ConSim.construct_prompt()`.

    Examples:
        Minimal runnable example with precomputed concept-explainer artifacts:

        >>> import torch
        >>> from interpreto.concepts.metrics import ConSim
        >>>
        >>> metric = ConSim(classes=["negative", "positive"])
        >>> samples = ["great acting", "boring plot", "loved the ending", "fell asleep"]
        >>> labels = torch.tensor([1, 0, 1, 0])
        >>> predictions = torch.tensor([1, 0, 0, 0])
        >>> # These artifacts usually come from a fitted concept explainer:
        >>> # - `concepts_interpretation` from TopKInputs(...) or LLMLabels(...)
        >>> # - `local_importances` from concept_explainer.concept_output_gradient(...)
        >>> concepts_interpretation = {
        ...     0: "positive sentiment words",
        ...     1: "negative sentiment words",
        ... }
        >>> global_importances = torch.tensor([
        ...     [-0.8, 0.7],
        ...     [0.9, -0.6],
        ... ])
        >>> local_importances = [
        ...     torch.tensor([[-0.2, 0.4], [0.5, -0.1]]),
        ...     torch.tensor([[0.3, -0.5], [-0.4, 0.2]]),
        ...     torch.tensor([[-0.1, 0.3], [0.4, -0.2]]),
        ...     torch.tensor([[0.5, -0.3], [-0.2, 0.1]]),
        ... ]
        >>> system_prompt, user_prompts, model_predictions = metric.construct_prompt(
        ...     setting=ConSim.prompt_types.E3_global_and_local_concepts_with_lp,
        ...     interesting_samples=samples,
        ...     corresponding_predictions=predictions,
        ...     corresponding_labels=labels,
        ...     nb_learning_samples=2,
        ...     concepts_interpretation=concepts_interpretation,
        ...     global_importances=global_importances,
        ...     local_importances=local_importances,
        ... )
        >>> len(system_prompt) > 0 and len(user_prompts) == len(model_predictions)
        True
        >>> # In practice, call your LLM interface here and pass its responses below.
        >>> responses = llm_interface.batch_generate(system_prompt, user_prompts)
        >>> metric.score_from_responses(responses, model_predictions)
    """

    prompt_types: type[PromptTypes] = PromptTypes

    @staticmethod
    def _quantize_importances(importance: float, threshold: float = 0.05) -> str | None:
        """
        Convert a normalized importance into the coarse symbols used in ConSim prompts.

        The mapping keeps prompts short and stable across explainers:
        `++` for values `>= 6 * threshold`, `+` for values `>= threshold`,
        `-` for values `<= -threshold`, `--` for values `<= -6 * threshold`,
        and `None` when the importance is too small to display.

        Arguments:
            importance: float
                Scalar importance value.
            threshold: float
                Minimum absolute magnitude required to display a concept.

        Returns:
            str | None:
                One of `{"++", "+", "-", "--"}` or `None` when the concept should be omitted.
        """
        if importance <= -6 * threshold:
            return "Very opposed"

        if importance <= -threshold:
            return "Opposed"

        if importance >= 6 * threshold:
            return "Highly supportive"

        if importance >= threshold:
            return "Supportive"

        return None

    @staticmethod
    def _resolve_prompt_setting(prompt_type: PromptTypes | PromptSetting) -> PromptSetting:
        """
        Normalize either a preset enum member or a direct `PromptSetting`.

        This lets public APIs accept the ergonomic preset form while keeping the internal renderer
        and validator focused on a single concrete type.
        """
        if isinstance(prompt_type, PromptTypes):
            return prompt_type.value
        return prompt_type

    @staticmethod
    def _concept_descriptor(
        concept_id: int,
        concepts_interpretation: dict[int, str],
    ) -> str:
        """
        Format one concept identifier for display in a prompt.

        Descriptions are truncated to 100 characters to keep prompts readable and to avoid a single
        verbose interpretation dominating the context window.
        """
        if concept_id not in concepts_interpretation:
            raise ValueError(f"Missing label for concept id {concept_id} in `concepts_interpretation`.")

        return f"C{concept_id} ({concepts_interpretation[concept_id][:100]})"

    @staticmethod
    def _concepts_to_string(
        importances: torch.Tensor,
        concepts_interpretation: dict[int, str],
        top_k: int = 5,
        threshold: float = 0.05,
    ) -> str:
        """
        Render a 1D concept-importance vector as a compact ConSim string.

        The method selects the top-`k` concepts by absolute magnitude, quantizes their values with
        `_quantize_importances`, and returns a short dictionary-like string such as
        `{C0 (token pattern): ++, C7 (named entity): -}`.

        Arguments:
            importances: torch.Tensor
                Concept scores, shape `(nb_concepts,)`.
            concepts_interpretation: dict[int, str]
                Human-readable label for each concept id.
            top_k: int
                Maximum number of concepts to inspect before thresholding.
            threshold: float
                Minimum absolute magnitude required for a concept to appear.

        Returns:
            str:
                Printable concept summary.
        """
        if not isinstance(importances, torch.Tensor) or importances.ndim != 1:
            raise ValueError("`importances` must be a 1D torch.Tensor of concept importances.")

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        k = min(top_k, importances.numel())
        top_indices = torch.topk(importances.abs(), k=k).indices.tolist()

        concepts_key_value = [
            (
                ConSim._concept_descriptor(concept_id, concepts_interpretation),
                ConSim._quantize_importances(importances[concept_id].item(), threshold=threshold),
            )
            for concept_id in top_indices
        ]

        return (
            "{"
            + ", ".join(
                [
                    f"{concept_interpretation}: {concept_importance}"
                    for concept_interpretation, concept_importance in concepts_key_value
                    if concept_importance is not None
                ]
            )
            + "}"
        )

    @staticmethod
    def _select_concept_ids(
        importances: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.05,
    ) -> set[int]:
        """
        Return the concept ids that would survive `_concepts_to_string`.

        This helper mirrors the rendering logic without building the final string, which is useful
        when a caller wants to track which concepts are exposed in a prompt.
        """
        if not isinstance(importances, torch.Tensor) or importances.ndim != 1:
            raise ValueError("`importances` must be a 1D torch.Tensor of concept importances.")

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        k = min(top_k, importances.numel())
        if k == 0:
            return set()

        top_indices = torch.topk(importances.abs(), k=k).indices.tolist()
        return {
            int(concept_id)
            for concept_id in top_indices
            if ConSim._quantize_importances(importances[concept_id].item(), threshold) is not None
        }

    @staticmethod
    def _setting_to_prompt(  # type: ignore[override]  # noqa: PLR0912  # ignore too many branches  # too many special cases
        setting: PromptSetting,
        interesting_samples: list[str],
        corresponding_predictions: torch.Tensor,
        corresponding_labels: torch.Tensor,
        nb_learning_samples: int,
        classes: dict[int, str],
        concepts_interpretation: dict[int, str],
        global_importances: dict[int, torch.Tensor],
        local_importances: list[torch.Tensor] | None,
        top_k: int = 5,
        importance_threshold: float = 0.05,
        contrastive_pairs: list[tuple[int, int]] | None = None,
    ) -> tuple[str, list[str], list[str]]:
        """
        Render a validated ConSim configuration into LLM-ready prompts.

        This is the low-level text renderer used by `construct_prompt`. Once inputs have been
        validated, it performs four deterministic steps:
            1. build the shared task description and class list;
            2. optionally add global concept summaries;
            3. optionally add learning-phase examples with local concept information;
            4. emit one user prompt per evaluation sample and the aligned expected class names.

        The method accepts more combinations than the predefined `PromptTypes`, because custom
        ablations may still be meaningful to power users.

        Arguments:
            setting: PromptSetting
                Resolved prompt configuration.
            interesting_samples: list[str]
                Selected samples, length `n_samples`.
                The first `nb_learning_samples` are used in the learning phase and the rest are evaluation samples.
            corresponding_predictions: torch.Tensor
                Model predictions aligned with `interesting_samples`, shape `(n_samples,)`.
            corresponding_labels: torch.Tensor
                Gold labels aligned with `interesting_samples`, shape `(n_samples,)`.
                Used by contrastive local prompts, with predictions as facts and labels as foils.
            nb_learning_samples: int
                Number of samples placed in the learning phase.
            classes: dict[int, str]
                Display names for the class ids present in this run.
            concepts_interpretation: dict[int, str]
                Human-readable label for each concept id.
            global_importances: dict[int, torch.Tensor]
                Mapping `class_id -> importance vector`, each vector with shape `(nb_concepts,)`.
            local_importances: list[torch.Tensor] | None
                Per-sample local importance tensors. Only the first `nb_learning_samples` entries
                are used; each entry must have shape `(nb_classes, nb_concepts)`.
            top_k: int
                Maximum number of concepts to inspect per rendered explanation.
            importance_threshold: float
                Minimum absolute magnitude required for a concept to appear.
            contrastive_pairs: list[tuple[int, int]] | None
                List of `(fact_class_id, foil_class_id)` pairs for contrastive global prompts.

        Returns:
            system_prompt: str
                Shared instructions and learning examples.
            user_prompts: list[str]
                One prompt per evaluation sample.
            literal_model_predictions: list[str]
                Expected class names aligned with `user_prompts`.
        """
        system_prompt_parts = []

        # ==============================================================================================
        # Global
        # ----------------
        # task description

        task_description_prompt = "You are a classifier. Your task is to assign a label to the evaluation sample. "
        if setting.global_contrastive_importances:
            task_description_prompt += "To complete the task, you will be given the concepts and their contrastive importance for each class. "
        elif setting.concepts_global_importances:
            task_description_prompt += (
                "To complete the task, you will be given the most important concepts for each class. "
            )
        if setting.lp_samples:
            if setting.lp_concepts_local_contributions:
                task_description_prompt += "You will have examples of samples, labels, and concepts contributions to labels as reference to learn the task. "
            elif setting.lp_local_contrastive_importance:
                task_description_prompt += "You will have examples of samples, labels, and contrastive concepts contributions (why predict this class and not the other) as reference to learn the task. "
            else:
                task_description_prompt += (
                    "You will have examples of samples and labels as reference to learn the task. "
                )
        if (
            setting.concepts_global_importances
            or setting.global_contrastive_importances
            or setting.lp_concepts_local_contributions
            or setting.lp_local_contrastive_importance
        ):
            task_description_prompt += " For each concept, the importances are 'Very opposed', 'Opposed', 'Supportive', or 'Highly supportive'. It means that a opposed concept is present in the text, the corresponding class is improbable. In the other hand, when a supportive concept is present in the text, the corresponding class is more likely."
        task_description_prompt += "User's prompt will contain an evaluation sample on which you should predict the class. Only return the class name, no other text."
        assert len(task_description_prompt) > 0
        system_prompt_parts.append(task_description_prompt)

        # -------
        # classes
        # if setting.pred_concepts:
        #     # show the concepts that could be predicted
        #     classes_prompt = f"The concepts are: [{', '.join(concepts_interpretation.keys())}]"
        if setting.anonymize_classes:
            classes = {i: f"Class_{i}" for i in classes.keys()}
        system_prompt_parts.append(f"The classes are: [{', '.join(list(classes.values()))}]")

        # ---------------------------
        # classes concepts importance
        if setting.concepts_global_importances:
            # for each anonymized class, show the top k concepts
            classes_concepts_prompt = (
                "The most important concepts and their importance for each class are:\n"
                + "\n".join(
                    [
                        # f"\t{class_name}: {
                        #     ConSim._concepts_to_string(
                        #         global_importances[class_index],
                        #         concepts_interpretation,
                        #         top_k=top_k,
                        #         threshold=importance_threshold,
                        #     )
                        # }"
                        f"\t{class_name}: {
                            ConSim._concepts_to_string(
                                global_importances[class_index],
                                concepts_interpretation,
                                top_k=top_k,
                                threshold=importance_threshold,
                            )
                        }"
                        for class_index, class_name in classes.items()
                    ]
                )
            )
            system_prompt_parts.append(classes_concepts_prompt)

        # global contrastive explanation
        if setting.global_contrastive_importances:
            if contrastive_pairs is None:
                raise ValueError(
                    "PromptSetting.global_contrastive_importances=True requires `contrastive_pairs` to be provided to ConSim.construct_prompt()."
                )

            contrastive_prompt_parts = []
            # for each contrastive pair, show the concept for fact - foil
            for pair in contrastive_pairs:
                contrastive_importance = global_importances[pair[0]] - global_importances[pair[1]]
                str_concept = ConSim._concepts_to_string(
                    contrastive_importance, concepts_interpretation, top_k=top_k, threshold=importance_threshold
                )
                contrastive_prompt_parts.append(f"\tfact: {classes[pair[0]]}, foil: {classes[pair[1]]}: {str_concept}")

            contrastive_global_prompt = (
                "The contrastively important concepts to choose fact over foil are:\n"
                + "\n".join(contrastive_prompt_parts)
            )
            system_prompt_parts.append(contrastive_global_prompt)

        # ==============================================================================================
        # Learning phase
        if setting.lp_samples:
            learning_phase_blocks = []
            for i in range(nb_learning_samples):
                # ----------------------------------------
                # samples text and label (predicted class)
                block = [
                    f"Sample_{i}:",
                    f"\tText: {interesting_samples[i]}",
                    f"\tLabel: {classes[int(corresponding_predictions[i])]}",
                ]

                # ----------------------------
                # concepts local contributions
                if setting.lp_concepts_local_contributions:
                    pred = int(corresponding_predictions[i].item())
                    str_importances = ConSim._concepts_to_string(
                        local_importances[i][pred],  # type: ignore
                        concepts_interpretation,
                        top_k=top_k,
                        threshold=importance_threshold,
                    )
                    block.append(f"\tConcepts contributions: {str_importances}")  # type: ignore

                # ----------------------------------------
                # contrastive local concepts contributions
                if setting.lp_local_contrastive_importance:
                    pred_index = int(corresponding_predictions[i].item())
                    gold_index = int(corresponding_labels[i].item())

                    # show the contrastive only for misclassified samples
                    if pred_index == gold_index:
                        text = "Concepts contributions"
                        importances = local_importances[i][pred_index]  # type: ignore
                    else:
                        text = f"Concepts contributions supporting {classes[pred_index]} rather than {classes[gold_index]}"
                        importances = local_importances[i][pred_index] - local_importances[i][gold_index]  # type: ignore

                    # convert the importances to a string
                    str_importances = ConSim._concepts_to_string(
                        importances,
                        concepts_interpretation,
                        top_k=top_k,
                        threshold=importance_threshold,
                    )
                    block.append(f"\t{text}: {str_importances}")  # type: ignore

                learning_phase_blocks.append("\n".join(block))
            system_prompt_parts.append("\n".join(learning_phase_blocks))

        # concatenate prompts parts
        system_prompt = "\n\n".join(system_prompt_parts)

        # ==============================================================================================
        # Inference
        # ----------------
        # show the samples
        user_prompts = [
            "\n".join(
                [
                    "Evaluation sample:",
                    f"\tText: {interesting_samples[i]}",
                    "\tLabel: ",
                ]
            )
            for i in range(nb_learning_samples, len(interesting_samples))
        ]

        # -----------------
        # model predictions (not included in the prompt, but returned to compute accuracy)
        literal_model_predictions = [
            classes[int(corresponding_predictions[i])] for i in range(nb_learning_samples, len(interesting_samples))
        ]

        return system_prompt, user_prompts, literal_model_predictions

    def _check_input_settings_correspondence(
        self,
        interesting_samples: list[str],
        corresponding_predictions: torch.Tensor,
        corresponding_labels: torch.Tensor,
        concepts_interpretation: dict[int, str],
        global_importances: torch.Tensor,
        nb_learning_samples: int,
        local_importances: list[torch.Tensor] | None,
        prompt_type: PromptTypes | PromptSetting = PromptTypes.E3_global_and_local_concepts_with_lp,
    ):
        """
        Validate that the selected samples and explanation tensors match the chosen setting.

        This method centralizes all shape checks and setting-dependent requirements before prompt
        rendering. Keeping validation separate from `_setting_to_prompt` makes failures easier to
        diagnose and keeps the renderer mostly focused on text generation.

        Arguments:
            interesting_samples: list[str]
                Selected samples, length `n_samples`.
            corresponding_predictions: torch.Tensor
                Model predictions aligned with `interesting_samples`, shape `(n_samples,)`.
            corresponding_labels: torch.Tensor
                Gold labels aligned with `interesting_samples`, shape `(n_samples,)`.
                The tensor is always required in the signature so callers do not need different
                code paths for contrastive versus non-contrastive settings.
            concepts_interpretation: dict[int, str]
                Human-readable label for each concept id.
            global_importances: torch.Tensor
                Global concept importances, shape `(len(self.classes), nb_concepts)`.
            nb_learning_samples: int
                Number of samples that will be placed in the learning phase.
            local_importances: list[torch.Tensor] | None
                Optional local concept importances. If provided, the list must contain at least
                `nb_learning_samples` entries and each entry must have shape
                `(len(self.classes), nb_concepts)`.
            prompt_type: PromptTypes | PromptSetting
                Either a preset from `PromptTypes` or a custom setting.

        Raises:
            ValueError:
                If shapes are inconsistent or a required explanation tensor is missing for the
                chosen setting.
        """
        setting = ConSim._resolve_prompt_setting(prompt_type)
        setting.validate(concepts_interpretation=concepts_interpretation, labels=corresponding_labels)

        # ==========================================================================================
        # Extensive checks
        if nb_learning_samples >= len(interesting_samples):
            raise ValueError(
                f"`nb_learning_samples` must be smaller than the number of samples. "
                f"Got {nb_learning_samples=} and {len(interesting_samples)=}."
            )

        if len(corresponding_predictions) != len(interesting_samples):
            raise ValueError(
                f"`corresponding_predictions` and `interesting_samples` must have the same length. "
                f"Got {len(corresponding_predictions)=} and {len(interesting_samples)=}."
            )

        if len(corresponding_labels) != len(interesting_samples):
            raise ValueError(
                f"`corresponding_labels` and `interesting_samples` must have the same length. "
                f"Got {len(corresponding_labels)=} and {len(interesting_samples)=}."
            )

        if not isinstance(global_importances, torch.Tensor) or global_importances.ndim != 2:
            raise ValueError("`global_importances` must be a torch.Tensor with shape (nb_classes, nb_concepts).")

        if global_importances.shape[0] != len(self.classes):
            raise ValueError(
                "`global_importances` must have shape (nb_classes, nb_concepts) with nb_classes matching `classes`."
            )

        if local_importances is not None:
            if not isinstance(local_importances, list) or not local_importances:
                raise ValueError(
                    "`local_importances` must be a non-empty list of torch.Tensor with shape (nb_classes, nb_concepts)."
                )
            if len(local_importances) < nb_learning_samples:
                raise ValueError(
                    "`local_importances` must have at least `nb_learning_samples` entries. "
                    f"Got {len(local_importances)=} and {nb_learning_samples=}"
                )
            for sample_importances in local_importances:
                if not isinstance(sample_importances, torch.Tensor) or sample_importances.ndim != 2:
                    raise ValueError(
                        "`local_importances` must be a list of torch.Tensor with shape (nb_classes, nb_concepts)."
                    )
                if sample_importances.shape[0] != len(self.classes):
                    raise ValueError(
                        "`local_importances` entries must have shape (nb_classes, nb_concepts) with nb_classes matching `classes`."
                    )
        elif setting.lp_concepts_local_contributions or setting.lp_local_contrastive_importance:
            raise ValueError(
                "PromptSetting.lp_concepts_local_contributions or PromptSetting.lp_local_contrastive_importance "
                "requires `local_importances` to be provided to ConSim.construct_prompt()."
            )

    def construct_prompt(  # type: ignore[override]
        self,
        setting: PromptTypes | PromptSetting,
        interesting_samples: list[str],
        corresponding_predictions: torch.Tensor,
        corresponding_labels: torch.Tensor,
        nb_learning_samples: int,
        *,
        concepts_interpretation: dict[int, str],
        global_importances: torch.Tensor,
        local_importances: list[torch.Tensor] | None = None,
        top_k: int = 5,
        importance_threshold: float = 0.05,
        contrastive_pairs: list[tuple[int, int]] | None = None,
    ) -> tuple[str, list[str], list[str]]:
        """
        Build the prompts needed to run a ConSim evaluation.

        It resolves the chosen prompt preset, validates all inputs, filters class-dependent
        artifacts to the classes that actually appear in `corresponding_predictions`, and delegates
        the final text rendering to `_setting_to_prompt`.

        The returned prompts follow the ConSim structure:
            - task description and class list;
            - optional global concept summaries;
            - optional learning-phase examples;
            - one user prompt per evaluation sample.

        Arguments:
            setting: PromptTypes | PromptSetting
                Preset or custom prompt configuration.
            interesting_samples: list[str]
                Selected samples, shape `(n_samples,)`. The first `nb_learning_samples` are used
                as learning-phase examples, the rest are evaluation samples.
            corresponding_predictions: torch.Tensor
                Model predictions aligned with `interesting_samples`, shape `(n_samples,)`.
            corresponding_labels: torch.Tensor
                Gold labels aligned with `interesting_samples`, shape `(n_samples,)`.
            nb_learning_samples: int
                Number of samples assigned to the learning phase. Must be smaller than
                `len(interesting_samples)`.
            concepts_interpretation: dict[int, str]
                Human-readable label for each concept id.
            global_importances: torch.Tensor
                Global concept importances, shape `(len(self.classes), nb_concepts)`.
            local_importances: list[torch.Tensor] | None
                Optional local concept importances. Required by settings that expose local concepts.
                Each entry must have shape `(len(self.classes), nb_concepts)`.
            top_k: int
                Maximum number of concepts to inspect per rendered explanation.
            importance_threshold: float
                Minimum absolute magnitude required for a concept to appear.
            contrastive_pairs: list[tuple[int, int]] | None
                List of `(fact_class_id, foil_class_id)` pairs for contrastive global prompts.

        Returns:
            system_prompt: str
                Shared instructions and learning examples.
            user_prompts: list[str]
                One prompt per evaluation sample.
            model_predictions: list[str]
                Expected class names aligned with `user_prompts`.

        Raises:
            ValueError
                If the chosen setting is incompatible with the provided inputs.

        Notes:
            Only classes present in `corresponding_predictions` are rendered into the prompt.
            When `contrastive_pairs` is provided, pairs involving absent classes are discarded.
        """
        setting = ConSim._resolve_prompt_setting(setting)

        # check inputs with respect to setting
        self._check_input_settings_correspondence(
            interesting_samples=interesting_samples,
            corresponding_predictions=corresponding_predictions,
            corresponding_labels=corresponding_labels,
            concepts_interpretation=concepts_interpretation,
            global_importances=global_importances,
            local_importances=local_importances,
            prompt_type=setting,
            nb_learning_samples=nb_learning_samples,
        )

        # extract the classes present in the predictions or the gold labels
        classes_ids = sorted(corresponding_predictions.unique().tolist())
        classes = {class_id: self.classes[class_id] for class_id in classes_ids}

        # filter based on classes subset
        global_importances_dict = {class_id: global_importances[class_id] for class_id in classes_ids}
        if contrastive_pairs is not None:
            contrastive_pairs = [
                pair for pair in contrastive_pairs if (pair[0] in classes_ids and pair[1] in classes_ids)
            ]  # type: ignore

        # integrate the different elements into a prompt
        return ConSim._setting_to_prompt(
            setting=setting,
            interesting_samples=interesting_samples,
            corresponding_predictions=corresponding_predictions,
            corresponding_labels=corresponding_labels,
            nb_learning_samples=nb_learning_samples,
            classes=classes,
            concepts_interpretation=concepts_interpretation,
            global_importances=global_importances_dict,
            local_importances=local_importances,
            top_k=top_k,
            importance_threshold=importance_threshold,
            contrastive_pairs=contrastive_pairs,
        )
