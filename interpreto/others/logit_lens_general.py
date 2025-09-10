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

# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL and FOR are research programs operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================

"""
General implementation of the Logit Lens method that adapts to different model types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import HTML, display
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    BatchEncoding,
    PreTrainedTokenizer,
)
from transformers.models.auto import modeling_auto

from interpreto.model_wrapping.model_with_split_points import ModelWithSplitPoints


class BaseLogitLens(ABC):
    """
    Base class for Logit Lens implementations.

    This class defines the common interface and shared functionality for different
    types of logit lens implementations (Language Model vs Classification).
    """

    def __init__(
        self,
        model: ModelWithSplitPoints,
        tokenizer: PreTrainedTokenizer,
        head_name: str | None = None,
        model_head: nn.Module | None = None,
        features_dim: int | None = None,
        normalization: bool = False,
        normalization_method: nn.Module | None = nn.LayerNorm,
        device: torch.device | None = None,
        batch_size: int = 8,
    ):
        # Check for meta tensors and reload model directly if needed
        self.splitted_model = model
        self.model = model._model
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer

        self.model_head = None
        self.activations = None
        self.head_name = head_name
        self.normalization = normalization
        self.normalization_method = normalization_method
        self.layer_names = None
        self.batch_size = batch_size

        # Initialize model head and features dimension
        self._initialize_head_and_features(features_dim)
        self.model_head = model_head.to(self.device) if model_head is not None else self.model_head

        if self.needs_model_reload():
            print("⚠️ Head access failed, likely due to meta tensors in model.")
            print("💡 This happens when you use ModelWithSplitPoints with automatic loading.")
            print("🔄 Reloading the model directly to get proper weights for the model head...")

            # Get the model path/repo_id from the original model
            model_path = self._get_model_or_repo_id()

            # Determine the appropriate model class and load directly
            model_autoclass = self.splitted_model.model_autoclass
            target_device = self.device

            if model_autoclass == modeling_auto.AutoModelForCausalLM:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map=target_device)
            elif model_autoclass == modeling_auto.AutoModelForMaskedLM:
                self.model = AutoModelForMaskedLM.from_pretrained(model_path, device_map=target_device)
            elif model_autoclass == modeling_auto.AutoModelForSequenceClassification:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map=target_device)
            else:
                # Fallback to AutoModel for other cases
                self.model = AutoModel.from_pretrained(model_path, device_map=target_device)

            print(f"✅ Successfully reloaded model directly with proper weights on {target_device}.")
            # Re-initialize model head
            self.head_name = head_name
            self._initialize_head_and_features(features_dim)

        if self.needs_model_reload(verbose=True):
            raise RuntimeError(
                "Failed to access model head even after reloading the model. Please check the model and head configuration."
            )

    def needs_model_reload(self, verbose: bool = False) -> bool:
        """Test if head access actually fails"""
        try:
            # Try to access and use the model head
            test_tensor = torch.randn(1, 10, self.features_dim).to(self.device)
            model_head = self.model_head.to(self.device)
            H = model_head(test_tensor)
            if torch.all(H < 1e-14) and torch.all(H > -1e-14):
                if verbose:
                    print("Model head returned all-zero output, indicating potential meta tensor issue.")
                return True  # Head likely has meta tensors
            return False  # Head works fine
        except Exception as e:
            if verbose:
                print(f"Model head access failed: {e}")
            return True  # Head access failed

    def _get_model_or_repo_id(self) -> str:
        """
        Retrieve the original model_or_repo_id parameter.

        Returns:
            str: The repository ID or model path
        """
        # Try to get repo_id from the model
        if hasattr(self.splitted_model, "repo_id") and self.splitted_model.repo_id:
            return self.splitted_model.repo_id

        # Fallback to model config name_or_path
        if hasattr(self.model, "config") and hasattr(self.model.config, "name_or_path"):
            return self.model.config.name_or_path

        # If all else fails, return a default indication
        return "unknown_model_path"

    def has_meta_tensors(self) -> bool:
        """
        Check if the model contains meta tensors.

        Returns:
            bool: True if meta tensors are detected, False otherwise.
        """
        # Check main model
        try:
            for param in self.model.parameters():
                if param.device.type == "meta":
                    return True
        except Exception:
            print("Warning: Could not check for meta tensors.")
            pass

        return False

    def _initialize_head_and_features(self, features_dim: int | None = None):
        """Initialize the model head and determine features dimension."""
        # Define possible head names for different model types
        possible_heads = [
            "lm_head",
            "cls.predictions.decoder",
            "cls.predictions",
            "cls",
            "score",
            "predictions",
            "decoder",
            "classifier",
            "head",
            "output",
        ]

        k = 0
        # Determine model head
        if self.head_name is None:
            print("No head name specified, trying to find a suitable head in the model.")
            for head_name in possible_heads:
                try:
                    head = self.model
                    for part in head_name.split("."):
                        head = getattr(head, part)

                    if isinstance(head, nn.Linear):
                        self.model_head = head
                        self.head_name = head_name
                        k += 1
                        break
                    elif isinstance(head, nn.Module):
                        if callable(head):
                            self.model_head = head
                            self.head_name = head_name
                            k += 1
                            break
                        sub_heads = [attr for attr in dir(head) if not attr.startswith("_")]
                        for sub_head_name in sub_heads:
                            sub_head = getattr(head, sub_head_name)
                            if isinstance(sub_head, nn.Linear):
                                self.model_head = sub_head
                                self.head_name = f"{head_name}.{sub_head_name}"
                                k += 1
                                break
                        if k > 0:
                            break
                except AttributeError:
                    continue
        elif hasattr(self.model, self.head_name):
            self.model_head = getattr(self.model, self.head_name)
            k += 1
            if not isinstance(self.model_head, nn.Module):
                raise ValueError(f"The pre-set head '{self.head_name}' is not a valid nn.Module.")
        else:
            raise ValueError(f"The specified head '{self.head_name}' does not exist in the model.")

        if k == 0:
            raise ValueError(
                "No known head found in the model. Please specify a valid head name or ensure the model has a compatible head."
            )
        elif k > 1:
            raise ValueError(
                f"Multiple heads found in the model. Please specify a valid head name. Found: {k} possible heads."
            )

        # Determine features dimension
        if features_dim is not None:
            self.features_dim = features_dim
            print(f"Using manually specified features dimension: {self.features_dim}")
        else:
            self.features_dim = self._auto_detect_features_dim()
            if self.features_dim is None:
                raise ValueError(
                    f"Could not automatically determine the features dimension for head '{self.head_name}'. "
                    f"Please provide the 'features_dim' parameter manually."
                )

    def _auto_detect_features_dim(self) -> int | None:
        """Auto-detect the features dimension from the model head."""
        if self.model_head:
            if hasattr(self.model_head, "in_features"):
                return self.model_head.in_features
            elif hasattr(self.model_head, "decoder") and hasattr(self.model_head.decoder, "in_features"):
                return self.model_head.decoder.in_features
            elif hasattr(self.model_head, "predictions") and hasattr(self.model_head.predictions, "decoder"):
                if hasattr(self.model_head.predictions.decoder, "in_features"):
                    return self.model_head.predictions.decoder.in_features
            else:
                for _name, module in self.model_head.named_modules():
                    if isinstance(module, nn.Linear) and hasattr(module, "in_features"):
                        return module.in_features
        return None

    @abstractmethod
    def do_lens(self, activation: torch.Tensor, inputs_model=None) -> torch.Tensor:
        """
        Apply the Logit Lens method to the model activations.

        Args:
            activation: Model activations tensor
            inputs_model: Optional model inputs for accessing attention mask

        Returns:
            Processed logits tensor
        """
        pass

    def _explain(
        self, inputs: str | list[str] | BatchEncoding | torch.Tensor, layers_name: str | list[str] | None = None
    ):
        """Generate explanations using the Logit Lens method."""
        if isinstance(inputs, (str, list)):
            inputs_model = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
            model_input_keys = self.model.forward.__code__.co_varnames
            for key in list(inputs_model.keys()):
                if key not in model_input_keys:
                    inputs_model.pop(key)
        else:
            inputs_model = inputs
        self.activations = self.splitted_model.get_activations(
            inputs_model, ModelWithSplitPoints.activation_granularities.ALL
        )

        if layers_name is None:
            layers_name = list(self.activations.keys())

        for layer in layers_name:
            if layer not in self.activations:
                raise ValueError(f"Layer '{layer}' not found in the model activations.")
        # Validate activations
        self._validate_activations(layers_name)
        if isinstance(layers_name, str):
            layers_name = [layers_name]

        self.layer_names = layers_name

        # Process activations through the lens
        logits_dict = {}
        for layer in layers_name:
            logits_dict[layer] = self.do_lens(self.activations[layer], inputs_model)
        return self._process_logits(logits_dict, inputs_model)

    def _safe_item(self, tensor):
        """Safely extract item from tensor, handling meta tensors."""
        if tensor.device.type == "meta":
            raise RuntimeError("Cannot extract item from meta tensor. Please ensure your model is properly loaded.")
        return tensor.detach().cpu().item()

    def _safe_tensor_to_cpu(self, tensor):
        """Safely move tensor to CPU, handling meta tensors."""
        if tensor.device.type == "meta":
            raise RuntimeError("Cannot move meta tensor to CPU. Please ensure your model is properly loaded.")
        return tensor.detach().cpu()

    @abstractmethod
    def _validate_activations(self, layers_name: list[str]):
        """Validate that activations have the expected shape for this model type."""
        pass

    @abstractmethod
    def _process_logits(self, logits_dict: dict[str, torch.Tensor], inputs_model) -> dict:
        """Process the logits according to the specific model type (LM vs Classification)."""
        pass

    @abstractmethod
    def _get_output_labels(self) -> dict[int, str]:
        """Get the labels for the model outputs (tokens for LM, classes for classification)."""
        pass

    def explain(
        self, inputs: str | list[str] | BatchEncoding | torch.Tensor, layers_name: str | list[str] | None = None
    ):
        """Generate explanations with batching support."""
        # Handle empty inputs
        if isinstance(inputs, str) and inputs.strip() == "":
            raise ValueError("Empty string input is not supported")
        elif isinstance(inputs, list) and len(inputs) == 0:
            raise ValueError("Empty list input is not supported")
        elif isinstance(inputs, torch.Tensor) and inputs.numel() == 0:
            raise ValueError("Empty tensor input is not supported")
        elif isinstance(inputs, BatchEncoding) and inputs["input_ids"].numel() == 0:
            raise ValueError("Empty BatchEncoding input is not supported")

        if isinstance(inputs, (str, list)):
            if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id < 0:
                print("Tokenizer does not have a padding token. Setting a default padding token.")
                self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else "[PAD]"

            if isinstance(inputs, str):
                inputs = [inputs]
            batched_inputs = [inputs[i : i + self.batch_size] for i in range(0, len(inputs), self.batch_size)]
        elif isinstance(inputs, BatchEncoding):
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            batched_inputs = [
                BatchEncoding(
                    {
                        "input_ids": input_ids[i : i + self.batch_size],
                        "attention_mask": attention_mask[i : i + self.batch_size],
                    }
                )
                for i in range(0, input_ids.shape[0], self.batch_size)
            ]
        elif isinstance(inputs, torch.Tensor):
            batched_inputs = torch.split(inputs, self.batch_size, dim=0)
        else:
            raise ValueError("Unsupported type for inputs")

        if layers_name is None:
            print("Producing lens prediction for all activations.")

        merged_results = {}
        for batch in batched_inputs:
            batch_results = self._explain(batch, layers_name)
            for layer, data in batch_results.items():
                if layer not in merged_results:
                    merged_results[layer] = data
                else:
                    # Merge batch results
                    self._merge_batch_results(merged_results[layer], data)
        return merged_results

    @abstractmethod
    def _merge_batch_results(self, existing_data: dict, new_data: dict):
        """Merge results from different batches."""
        pass

    def __call__(
        self, inputs: str | list[str] | BatchEncoding | torch.Tensor, layers_name: str | list[str] | None = None
    ):
        """Enable the instance to be called as a function."""
        return self.explain(inputs, layers_name)


class LanguageModelLogitLens(BaseLogitLens):
    """
    Logit Lens implementation for Language Models (Causal LM and Masked LM).

    This implementation shows what tokens the model "thinks" should come next
    or fill a mask at each layer.
    """

    def __init__(self, model: ModelWithSplitPoints, tokenizer: PreTrainedTokenizer, nb_token: int = 5, **kwargs):
        self.vocab_size = tokenizer.vocab_size if tokenizer else None
        self.nb_token = min(nb_token, self.vocab_size) if self.vocab_size else nb_token
        super().__init__(model, tokenizer, **kwargs)

    def do_lens(self, activation: torch.Tensor, inputs_model=None) -> torch.Tensor:
        """
        Apply the Logit Lens method to language model activations.

        Args:
            activation: Model activations tensor of shape (batch_size, seq_len, hidden_dim)
            inputs_model: Not used for language models

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        if not self.model_head:
            raise ValueError("Model head is not set.")

        if self.normalization:
            activation = self.normalization_method(activation.size(-1))(activation)

        activation = activation.to(self.device)
        logits = self.model_head(activation)

        # Handle cases where model head returns tuple
        if isinstance(logits, tuple):
            logits = logits[0]  # The first element should be the logits
        return logits

    def _validate_activations(self, layers_name: list[str]):
        """Validate activations for language models - should match features_dim."""
        for layer in layers_name:
            activation = self.activations[layer]
            if activation.shape[-1] != self.features_dim:
                raise ValueError(
                    f"Activation shape mismatch for layer '{layer}': expected {self.features_dim}, got {activation.shape[-1]}."
                )

    def _process_logits(self, logits_dict: dict[str, torch.Tensor], inputs_model) -> dict:
        """Process logits for language modeling - convert to probabilities and get top-k tokens."""
        if self.nb_token == 0:
            print("WARNING: `nb_token` is set to 0, no top-k tokens will be returned.")
            return {layer: F.softmax(logits, dim=-1) for layer, logits in logits_dict.items()}

        results = {}
        for layer, logits in logits_dict.items():
            proba = F.softmax(logits, dim=-1)
            top_k_indices = torch.topk(proba, self.nb_token, dim=-1).indices

            results[layer] = {
                "tokens": np.array(
                    [
                        [
                            [
                                self.tokenizer.decode([self._safe_item(index)])
                                for index in top_k_indices[batch_idx, seq_idx]
                            ]
                            for seq_idx in range(top_k_indices.shape[1])
                        ]
                        for batch_idx in range(top_k_indices.shape[0])
                    ]
                ),
                "proba": np.array(
                    [
                        [
                            proba[batch_idx, seq_idx, top_k_indices[batch_idx, seq_idx]].detach().cpu().numpy()
                            for seq_idx in range(top_k_indices.shape[1])
                        ]
                        for batch_idx in range(top_k_indices.shape[0])
                    ]
                ),
            }
        return results

    def _get_output_labels(self) -> dict[int, str]:
        """Get token labels from tokenizer vocabulary."""
        return {i: self.tokenizer.decode([i]) for i in range(self.vocab_size)}

    def _merge_batch_results(self, existing_data: dict, new_data: dict):
        """Merge token results from different batches."""
        # Handle empty data cases
        if existing_data["tokens"].size == 0:
            existing_data.update(new_data)
            return
        if new_data["tokens"].size == 0:
            return

        max_seq_len = max(existing_data["tokens"].shape[1], new_data["tokens"].shape[1])
        pad_token = self.tokenizer.pad_token if self.tokenizer.pad_token is not None else "<pad>"

        # Padding for each dimension
        existing_pad = max_seq_len - existing_data["tokens"].shape[1]
        new_pad = max_seq_len - new_data["tokens"].shape[1]

        # Tokens: (batch_size, seq_len, nb_token) - pad seq_len dimension only
        merged_tokens = np.concatenate(
            [
                np.pad(
                    existing_data["tokens"],
                    ((0, 0), (0, existing_pad), (0, 0)),
                    mode="constant",
                    constant_values=pad_token,
                ),
                np.pad(new_data["tokens"], ((0, 0), (0, new_pad), (0, 0)), mode="constant", constant_values=pad_token),
            ],
            axis=0,
        )

        # Proba: (batch_size, seq_len, nb_token) - pad seq_len dimension only
        merged_proba = np.concatenate(
            [
                np.pad(
                    existing_data["proba"], ((0, 0), (0, existing_pad), (0, 0)), mode="constant", constant_values=0.0
                ),
                np.pad(new_data["proba"], ((0, 0), (0, new_pad), (0, 0)), mode="constant", constant_values=0.0),
            ],
            axis=0,
        )

        existing_data["tokens"] = merged_tokens
        existing_data["proba"] = merged_proba

    def visualize_logit_lens_interactive(
        self, inputs: str | list[str] | "BatchEncoding" | "torch.Tensor", layers_name: str | list[str] | None = None
    ):
        """
        Interactive HTML/JS visualization for Logit Lens predictions.
        Each input token is shown as plain text (tokenization artifacts prettified).
        On mouseover, a tooltip shows the top-k predicted tokens and their probabilities
        (probabilities are visible and color-coded).
        """

        def _prob_to_color(prob, min_prob=0.0, max_prob=1.0):
            prob = float((prob - min_prob) / (max_prob - min_prob))
            r = int(255 * prob)
            g = 0
            b = int(255 * (1 - prob))
            return f"rgb({r},{g},{b})"

        def _escape_html(text):
            return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

        def _prettify_token(token: str) -> str:
            """
            Clean up tokenization artifacts to make tokens more readable.

            Args:
                token: Raw token from tokenizer

            Returns:
                Cleaned token with artifacts removed
            """
            # Handle different tokenization schemes
            cleaned = token

            # GPT-style: Ġ represents spaces
            cleaned = cleaned.replace("Ġ", " ")

            # SentencePiece-style: ▁ represents spaces
            cleaned = cleaned.replace("▁", " ")

            # End-of-word markers
            cleaned = cleaned.replace("</w>", "")

            # BERT-style: ## prefix indicates continuation of previous word
            if cleaned.startswith("##"):
                cleaned = cleaned[2:]  # Remove ## prefix

            return cleaned

        def _group_bert_tokens(tokens: list[str]) -> list[str]:
            """
            Group BERT tokens that belong to the same word for better readability.

            Args:
                tokens: List of raw tokens from BERT tokenizer

            Returns:
                List of grouped tokens with better visual representation
            """
            if not tokens:
                return tokens

            grouped = []
            current_word_parts = []

            for token in tokens:
                if token.startswith("##"):
                    # This is a continuation of the previous word
                    current_word_parts.append(token[2:])  # Remove ##
                else:
                    # This starts a new word
                    if current_word_parts:
                        # Finish the previous word
                        grouped.append("".join(current_word_parts))
                        current_word_parts = []

                    # Start new word
                    if token in ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"]:
                        # Special tokens
                        grouped.append(f"⟨{token[1:-1]}⟩")
                    else:
                        current_word_parts = [token]

            # Don't forget the last word
            if current_word_parts:
                grouped.append("".join(current_word_parts))

            return grouped

        explaining = self.explain(inputs, layers_name)
        layers_name = self.layer_names if layers_name is None else layers_name

        if self.nb_token == 0:
            print(
                "WARNING: `nb_token` is set to 0, no top-k tokens will be returned. If you want to see the top-k tokens, please set `nb_token` to a positive integer."
            )
            return

        if isinstance(inputs, (str, list)):
            if isinstance(inputs, str):
                batch_tokens = [self.tokenizer.tokenize(inputs)]
            else:
                batch_tokens = [self.tokenizer.tokenize(s) for s in inputs]
        elif hasattr(inputs, "input_ids"):
            batch_tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in inputs["input_ids"]]
        elif hasattr(inputs, "cpu"):
            batch_tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in inputs.cpu().numpy()]
        else:
            raise ValueError("Unsupported type for inputs")

        nb_inputs = len(batch_tokens)
        html = """
        <style>
        .logitlens-token {
            display:inline-block;
            border-radius:3px;
            padding:2px 4px;
            margin:0 2px;
            cursor:pointer;
            background:none;
            color:inherit;
            font-family:inherit;
            font-size:inherit;
            transition: background 0.2s;
            position:relative;
        }
        .word-start {
            border-radius:3px 0px 0px 3px;
            background:rgba(220,220,220,0.2);
            border:1px solid #007acc;
            border-right:none;
            margin-right:0px !important;
        }
        .word-middle {
            border-radius:0px;
            background:rgba(220,220,220,0.2);
            border-top:1px solid #007acc;
            border-bottom:1px solid #007acc;
            margin-left:0px !important;
            margin-right:0px !important;
        }
        .word-end {
            border-radius:0px 3px 3px 0px;
            background:rgba(220,220,220,0.2);
            border:1px solid #007acc;
            border-left:none;
            margin-left:0px !important;
        }
        .logitlens-tooltip {
            display:none;
            position:absolute;
            background:rgba(255,255,255,0.98);
            border:1px solid #888;
            border-radius:6px;
            box-shadow:2px 2px 8px #aaa;
            padding:8px;
            font-size:13px;
            z-index:1000;
            max-width:330px;
            left:0;
            top:100%;
            white-space:normal;
            min-width:120px;
        }
        .logitlens-token:hover .logitlens-tooltip {
            display:block;
        }
        </style>
        <div>
        """

        for layer in layers_name:
            html += f"<div><strong>Layer: {layer}</strong></div>"
            tokens = batch_tokens[0] if nb_inputs == 1 else batch_tokens
            layer_data = explaining[layer]
            topk_tokens = layer_data["tokens"]
            topk_proba = layer_data["proba"]
            for batch_idx, sentence in enumerate(tokens if nb_inputs > 1 else [tokens]):
                html += "<div style='margin-bottom:12px;'>"

                # Check if this looks like BERT tokenization (has ## tokens)
                has_bert_tokens = any(token.startswith("##") for token in sentence)

                if has_bert_tokens:
                    # First, identify word boundaries
                    word_groups = []
                    current_group = []

                    for tok_idx, token in enumerate(sentence):
                        if token.startswith("##"):
                            # Continuation token - add to current group
                            current_group.append(tok_idx)
                        else:
                            # Start of new word - finish previous group and start new one
                            if current_group:
                                word_groups.append(current_group)
                            current_group = [tok_idx]

                    if current_group:
                        word_groups.append(current_group)

                    # Create a mapping of token index to word group info
                    token_to_group = {}
                    for group_idx, group in enumerate(word_groups):
                        for pos_in_group, tok_idx in enumerate(group):
                            token_to_group[tok_idx] = {
                                "group_id": group_idx,
                                "position": pos_in_group,
                                "group_size": len(group),
                                "is_first": pos_in_group == 0,
                                "is_last": pos_in_group == len(group) - 1,
                                "is_single": len(group) == 1,
                            }

                    for tok_idx, token in enumerate(sentence):
                        if tok_idx >= topk_tokens.shape[1]:
                            continue
                        tk_toks = topk_tokens[batch_idx, tok_idx]
                        tk_probs = topk_proba[batch_idx, tok_idx]
                        tooltip_html = "<div><b>Top-k predictions:</b><ul style='padding-left:16px;margin:0;'>"
                        for tk, prob in zip(tk_toks, tk_probs, strict=False):
                            color = _prob_to_color(prob)
                            tk_esc = _escape_html(_prettify_token(str(tk)))
                            tooltip_html += (
                                f"<li><span style='color:{color};font-weight:bold;'>{tk_esc}</span>: {prob:.3f}</li>"
                            )
                        tooltip_html += "</ul></div>"

                        # Improved token display for BERT
                        token_display = _prettify_token(str(token))
                        css_classes = ["logitlens-token"]
                        space_before = ""

                        if tok_idx in token_to_group:
                            group_info = token_to_group[tok_idx]

                            if group_info["is_single"]:
                                # Single token word - no special styling needed
                                space_before = " " if tok_idx > 0 else ""
                            # Multi-token word
                            elif group_info["is_first"]:
                                css_classes.append("word-start")
                                space_before = " " if tok_idx > 0 else ""
                            elif group_info["is_last"]:
                                css_classes.append("word-end")
                                space_before = ""
                            else:
                                css_classes.append("word-middle")
                                space_before = ""
                        else:
                            # Fallback for tokens not in groups
                            space_before = " " if tok_idx > 0 else ""

                        token_esc = _escape_html(token_display)
                        css_class_str = " ".join(css_classes)
                        html += (
                            f"{space_before}<span class='{css_class_str}'>{token_esc}"
                            f"<span class='logitlens-tooltip'>{tooltip_html}</span>"
                            "</span>"
                        )
                else:
                    # Original token display for non-BERT models
                    for tok_idx, token in enumerate(sentence):
                        if tok_idx >= topk_tokens.shape[1]:
                            continue
                        tk_toks = topk_tokens[batch_idx, tok_idx]
                        tk_probs = topk_proba[batch_idx, tok_idx]
                        tooltip_html = "<div><b>Top-k predictions:</b><ul style='padding-left:16px;margin:0;'>"
                        for tk, prob in zip(tk_toks, tk_probs, strict=False):
                            color = _prob_to_color(prob)
                            tk_esc = _escape_html(_prettify_token(str(tk)))
                            tooltip_html += (
                                f"<li><span style='color:{color};font-weight:bold;'>{tk_esc}</span>: {prob:.3f}</li>"
                            )
                        tooltip_html += "</ul></div>"
                        token_esc = _escape_html(_prettify_token(str(token)))
                        html += (
                            f"<span class='logitlens-token'>{token_esc}"
                            f"<span class='logitlens-tooltip'>{tooltip_html}</span>"
                            "</span> "
                        )
                html += "</div>"

        html += "</div>"
        display(HTML(html))

    def lens(self, inputs: str | list[str] | BatchEncoding | torch.Tensor, layers_name: str | list[str] | None = None):
        """
        Plot a readable lens visualization for language models.
        """

        if isinstance(layers_name, str):
            layers_name = [layers_name]

        if isinstance(inputs, list):
            pass  # List of strings input
        elif isinstance(inputs, str):
            pass  # Single string input
        elif isinstance(inputs, BatchEncoding):
            pass  # Batch encoding input
        else:
            pass  # Other input types

        if isinstance(inputs, torch.Tensor) and len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(0)
        elif isinstance(inputs, torch.Tensor) and len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(0).unsqueeze(0)

        explaining = self.explain(inputs, layers_name)
        layers_name = self.layer_names if layers_name is None else layers_name

        if self.nb_token == 0:
            print(
                "WARNING: `nb_token` is set to 0, no top-k tokens will be returned. If you want to see the top-k tokens, please set `nb_token` to a positive integer."
            )
            return explaining

        self.visualize_logit_lens_interactive(inputs, layers_name)


class ClassificationLogitLens(BaseLogitLens):
    """
    Logit Lens implementation for Classification Models.

    This implementation shows what classification decision the model is leaning
    towards at each layer.
    """

    def __init__(
        self,
        model: ModelWithSplitPoints,
        tokenizer: PreTrainedTokenizer,
        pooling_strategy: str = "cls",  # "cls", "mean", "last"
        **kwargs,
    ):
        self.pooling_strategy = pooling_strategy
        self.original_pooling_strategy = pooling_strategy  # Keep track of original strategy
        super().__init__(model, tokenizer, **kwargs)

        # Check if the head has its own pooling and adjust strategy accordingly
        self._detect_and_handle_head_pooling()

        # Get number of classes from the model head
        self.num_classes = self._get_num_classes()

    def _get_num_classes(self) -> int:
        """Get the number of classes from the model head."""
        # Common case: direct out_features attribute
        if hasattr(self.model_head, "out_features"):
            return self.model_head.out_features
        # Check for decoder submodule
        elif hasattr(self.model_head, "decoder") and hasattr(self.model_head.decoder, "out_features"):
            return self.model_head.decoder.out_features
        else:
            # Find all nn.Linear modules inside the head
            linear_modules = [
                module for name, module in self.model_head.named_modules() if isinstance(module, nn.Linear)
            ]
            if len(linear_modules) == 1:
                # Only one linear layer, use its out_features
                return linear_modules[0].out_features
            elif len(linear_modules) > 1:
                # More than one linear layer, pick the last one (usually the classifier output)
                return linear_modules[-1].out_features
        raise ValueError(
            "Could not determine number of classes from the model head. Please set the number of classes manually using `set_num_classes`."
        )

    def set_num_classes(self, num_classes: int):
        """Set the number of classes for the classification head."""
        self.num_classes = num_classes

    def _detect_and_handle_head_pooling(self):
        """
        Detect if the model head performs its own pooling and adjust the pooling strategy accordingly.
        This handles cases like RobertaClassificationHead that expect full sequences and pool internally.
        """
        if not self.model_head:
            return

        head_class_name = self.model_head.__class__.__name__.lower()

        # List of known head types that perform their own pooling
        pooling_head_patterns = ["classificationhead", "sequenceclassificationhead", "clshead", "classification_head"]

        # Check if head class name suggests it does its own pooling
        has_internal_pooling = any(pattern in head_class_name for pattern in pooling_head_patterns)

        # Additional check: Look for pooling-related attributes/modules in the head
        if not has_internal_pooling:
            for name, module in self.model_head.named_modules():
                if any(pooling_term in name.lower() for pooling_term in ["pool", "cls", "dense"]):
                    # Check if it's not just a simple linear layer
                    if not (isinstance(module, nn.Linear) and len(list(self.model_head.named_modules())) <= 2):
                        has_internal_pooling = True
                        break

        if has_internal_pooling:
            print(f"⚠️  Warning: Detected that {self.model_head.__class__.__name__} likely performs its own pooling.")
            print(
                f"   Disabling external pooling strategy (was: '{self.original_pooling_strategy}') to avoid conflicts."
            )
            print("   The head will receive the full sequence and handle pooling internally.")
            self.pooling_strategy = None
        else:
            # Check if head is complex but pooling strategy is enabled
            head_modules = list(self.model_head.named_modules())
            if (
                len(head_modules) > 2 and self.pooling_strategy is not None
            ):  # More than just the head itself and one linear layer
                print(
                    f"⚠️  Warning: {self.model_head.__class__.__name__} appears to be a complex head with multiple layers."
                )
                print(f"   Current pooling strategy: '{self.pooling_strategy}'")
                print(
                    "   If you encounter dimension mismatches, the head might be incompatible with external pooling."
                )
                print("   Consider setting pooling_strategy=None if issues arise.")

    def _get_sequence_representation(
        self, activation: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Extract sequence-level representation for classification.

        Args:
            activation: Tensor of shape (batch_size, seq_len, hidden_dim)
            attention_mask: Optional attention mask

        Returns:
            Tensor of shape (batch_size, hidden_dim) or (batch_size, seq_len, hidden_dim) if no pooling
        """
        if self.pooling_strategy is None:
            # No pooling - return the full sequence (head will handle it)
            return activation
        elif self.pooling_strategy == "cls":
            # Use [CLS] token (first token)
            return activation[:, 0, :]
        elif self.pooling_strategy == "last":
            # Use last non-padded token
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_size = activation.shape[0]
                return activation[range(batch_size), seq_lengths, :]
            else:
                return activation[:, -1, :]
        elif self.pooling_strategy == "mean":
            # Use mean pooling
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand(activation.size())
                masked_activation = activation * mask
                return masked_activation.sum(dim=1) / mask.sum(dim=1)
            else:
                return activation.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def do_lens(self, activation: torch.Tensor, inputs_model=None) -> torch.Tensor:
        """
        Apply the Logit Lens method to classification model activations.

        Args:
            activation: Model activations tensor of shape (batch_size, seq_len, hidden_dim)
            inputs_model: Model inputs containing attention_mask for proper pooling

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        if not self.model_head:
            raise ValueError("Model head is not set.")

        # If pooling_strategy is None, pass the full activation to the head (it will do its own pooling)
        if self.pooling_strategy is None:
            if activation.dim() == 2:
                # If activation is already pooled but head expects sequences, this might be an issue
                print(
                    f"⚠️  Warning: Activation is already pooled (shape: {activation.shape}) but head expects to do its own pooling."
                )
                print("   This might cause issues. Consider checking your model architecture.")

            # Pass the full activation (or pooled if that's what we have) to the head
            input_to_head = activation
        else:
            # Get attention mask from inputs if available
            attention_mask = None
            if inputs_model is not None:
                attention_mask = inputs_model.get("attention_mask", None)

            # Handle case where activation is already pooled (batch_size, hidden_dim)
            if activation.dim() == 2:
                input_to_head = activation
            elif activation.dim() == 3:
                # Get sequence-level representation using pooling
                input_to_head = self._get_sequence_representation(activation, attention_mask)
            else:
                raise ValueError(
                    f"Unexpected activation tensor shape: {activation.shape}. Expected 2D (batch_size, hidden_dim) or 3D (batch_size, seq_len, hidden_dim)."
                )

        if self.normalization:
            if input_to_head.dim() == 3:
                # Apply normalization to the last dimension for 3D tensors
                norm_layer = self.normalization_method(input_to_head.size(-1))
                input_to_head = norm_layer(input_to_head)
            else:
                # Apply normalization for 2D tensors
                norm_layer = self.normalization_method(input_to_head.size(-1))
                input_to_head = norm_layer(input_to_head)

        input_to_head = input_to_head.to(self.device)
        logits = self.model_head(input_to_head)

        # Handle cases where model head returns tuple
        if isinstance(logits, tuple):
            logits = logits[0]  # The first element should be the logits

        return logits

    def _validate_activations(self, layers_name: list[str]):
        """
        Validate activations for classification models.
        """
        for layer in layers_name:
            activation = self.activations[layer]
            if activation.dim() not in [2, 3]:
                raise ValueError(
                    f"Activation for layer '{layer}' must be 2D (batch_size, hidden_dim) or 3D (batch_size, seq_len, hidden_dim), got shape: {activation.shape}"
                )

            # For 3D activations, we can handle any hidden dimension via pooling
            # For 2D activations, they should match the model head's expected input dimension
            if activation.dim() == 2 and activation.shape[-1] != self.features_dim:
                raise ValueError(
                    f"2D activation for layer '{layer}' should match features_dim: expected {self.features_dim}, got {activation.shape[-1]}"
                )

    def _process_logits(self, logits_dict: dict[str, torch.Tensor], inputs_model) -> dict:
        """
        Process logits for classification - convert to probabilities and class predictions.

        Args:
            logits_dict: Dictionary mapping layer names to logits tensors (already processed by do_lens)
            inputs_model: The tokenized inputs

        Returns:
            Dictionary with processed results for each layer
        """
        results = {}
        for layer, logits in logits_dict.items():
            # Verify that logits have the expected shape for classification
            if logits.dim() != 2:
                raise ValueError(
                    f"Expected 2D logits tensor (batch_size, num_classes) for layer '{layer}', got shape: {logits.shape}"
                )

            if logits.shape[-1] != self.num_classes:
                raise ValueError(
                    f"Logits shape mismatch for layer '{layer}': expected {self.num_classes} classes, got {logits.shape[-1]}"
                )

            # Convert to probabilities
            proba = F.softmax(logits, dim=-1)

            # Get predicted classes
            predicted_classes = torch.argmax(proba, dim=-1)

            # Get top-k predictions
            top_k_values, top_k_indices = torch.topk(proba, k=min(self.num_classes, 5), dim=-1)

            # Get class labels
            class_labels = self._get_output_labels()

            # Format top-k predictions with labels
            batch_size = proba.shape[0]
            top_k_predictions = []
            for batch_idx in range(batch_size):
                batch_predictions = []
                for k_idx in range(top_k_indices.shape[-1]):
                    class_idx = self._safe_item(top_k_indices[batch_idx, k_idx])
                    class_prob = self._safe_item(top_k_values[batch_idx, k_idx])
                    class_label = class_labels.get(class_idx, f"Class_{class_idx}")
                    batch_predictions.append(
                        {"class_id": class_idx, "class_label": class_label, "probability": class_prob}
                    )
                top_k_predictions.append(batch_predictions)

            results[layer] = {
                "logits": logits.detach().cpu().numpy(),
                "probabilities": proba.detach().cpu().numpy(),
                "predicted_classes": predicted_classes.detach().cpu().numpy(),
                "predicted_labels": [
                    class_labels.get(self._safe_item(idx), f"Class_{self._safe_item(idx)}")
                    for idx in predicted_classes
                ],
                "top_k_predictions": top_k_predictions,
                "class_labels": class_labels,
                "confidence_scores": torch.max(proba, dim=-1)[0].detach().cpu().numpy(),
            }

        return results

    def _get_output_labels(self) -> dict[int, str]:
        """
        Get class labels for the model outputs.

        Returns:
            Dictionary mapping class indices to human-readable labels
        """
        # First, try to get labels from model config
        if hasattr(self.model, "config") and hasattr(self.model.config, "id2label"):
            return self.model.config.id2label

        # Check if the wrapped model has the config
        if hasattr(self.splitted_model, "_model") and hasattr(self.splitted_model._model, "config"):
            config = self.splitted_model._model.config
            if hasattr(config, "id2label") and config.id2label:
                return config.id2label

        # Try to infer common classification tasks based on number of classes
        if self.num_classes == 2:
            return {0: "Negative", 1: "Positive"}
        elif self.num_classes == 3:
            # Common for sentiment analysis
            return {0: "Negative", 1: "Neutral", 2: "Positive"}
        elif self.num_classes == 5:
            # Common for rating/sentiment (1-5 stars)
            return {i: f"{i + 1}_star" for i in range(5)}
        else:
            # Generic labels as fallback
            return {i: f"Class_{i}" for i in range(self.num_classes)}

    def _merge_batch_results(self, existing_data: dict, new_data: dict):
        """
        Merge classification results from different batches.

        Args:
            existing_data: Dictionary containing results from previous batches
            new_data: Dictionary containing results from current batch
        """
        for key, value in new_data.items():
            if key == "class_labels":
                # Class labels are the same across batches, no need to merge
                continue
            elif key == "top_k_predictions":
                # Merge list of lists
                existing_data[key].extend(value)
            elif key == "predicted_labels":
                # Merge list of strings
                existing_data[key].extend(value)
            elif isinstance(value, np.ndarray):
                # Merge numpy arrays along batch dimension
                existing_data[key] = np.concatenate([existing_data[key], value], axis=0)
            # For any other data types, try to extend if it's a list
            elif isinstance(existing_data[key], list):
                if isinstance(value, list):
                    existing_data[key].extend(value)
                else:
                    existing_data[key].append(value)
            else:
                # If we can't merge, keep the existing data
                pass

    def visualize_classification_lens(
        self, inputs: str | list[str] | BatchEncoding | torch.Tensor, layers_name: str | list[str] | None = None
    ):
        """
        Visualize classification logit lens results.

        Args:
            inputs: Input text or tokenized inputs
            layers_name: Specific layers to analyze
        """
        results = self.explain(inputs, layers_name)

        if isinstance(inputs, str):
            inputs = [inputs]
        elif isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], str):
            pass  # Already a list of strings
        else:
            # For tokenized inputs, we'll show generic input labels
            if hasattr(inputs, "input_ids"):
                batch_size = inputs["input_ids"].shape[0]
            else:
                batch_size = inputs.shape[0] if hasattr(inputs, "shape") else 1
            inputs = [f"Input {i + 1}" for i in range(batch_size)]

        # Create HTML visualization
        html = """
        <style>
        .classification-lens {
            font-family: Arial, sans-serif;
            margin: 10px 0;
        }
        .layer-section {
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
        }
        .layer-title {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }
        .input-section {
            margin: 10px 0;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        .input-text {
            font-weight: bold;
            color: #555;
            margin-bottom: 8px;
        }
        .prediction {
            margin: 5px 0;
            padding: 5px;
            border-radius: 3px;
        }
        .top-prediction {
            background-color: #e8f5e8;
            border-left: 4px solid #4caf50;
        }
        .other-prediction {
            background-color: #f0f0f0;
            border-left: 4px solid #999;
        }
        .confidence-bar {
            display: inline-block;
            height: 10px;
            background-color: #ddd;
            border-radius: 5px;
            margin-left: 10px;
            vertical-align: middle;
        }
        .confidence-fill {
            height: 100%;
            border-radius: 5px;
            background: linear-gradient(to right, #ff6b6b, #feca57, #48dbfb, #0abde3);
        }
        </style>
        <div class="classification-lens">
        """

        for layer_name, layer_results in results.items():
            html += '<div class="layer-section">'
            html += f'<div class="layer-title">Layer: {layer_name}</div>'

            # Process each input in the batch
            for input_idx, input_text in enumerate(inputs):
                html += '<div class="input-section">'
                html += f'<div class="input-text">Input: {input_text}</div>'

                # Get top predictions for this input
                top_predictions = layer_results["top_k_predictions"][input_idx]

                for pred_idx, prediction in enumerate(top_predictions):
                    class_id = prediction["class_id"]
                    class_label = prediction["class_label"]
                    probability = prediction["probability"]

                    pred_class = "top-prediction" if pred_idx == 0 else "other-prediction"

                    html += f'<div class="prediction {pred_class}">'
                    html += f"<strong>{class_label}</strong> (Class {class_id}): {probability:.3f}"

                    # Add confidence bar
                    bar_width = int(probability * 100)
                    html += '<div class="confidence-bar" style="width: 100px;">'
                    html += f'<div class="confidence-fill" style="width: {bar_width}%;"></div>'
                    html += "</div>"
                    html += "</div>"

                html += "</div>"  # Close input-section

            html += "</div>"  # Close layer-section

        html += "</div>"

        display(HTML(html))

    def lens(self, inputs: str | list[str] | BatchEncoding | torch.Tensor, layers_name: str | list[str] | None = None):
        """
        Convenient method to visualize classification lens results.
        """
        self.visualize_classification_lens(inputs, layers_name)


class LogitLens:
    """
    Factory class for creating LogitLens implementations.

    The Logit Lens technique is a mechanistic interpretability method that analyzes
    what a transformer model "thinks" at each layer by projecting intermediate
    activations through the model's final prediction head (e.g., language modeling head
    or classification head). This allows us to observe how the model's predictions
    evolve layer by layer, providing insights into the model's internal reasoning process.

    The technique works by:
    1. Running a forward pass through the model and collecting activations at each layer
    2. Taking these intermediate representations and passing them through the final
       prediction head (bypassing the remaining layers)
    3. Converting the resulting logits to probabilities to see what the model would
       predict if it stopped processing at that layer
    4. Visualizing how predictions change and confidence builds up across layers

    Originally introduced in "Interpreting GPT: the logit lens" by nostalgebraist:
    https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

    Automatically chooses the appropriate implementation based on the model type:
    - **Language Models** (AutoModelForCausalLM, AutoModelForMaskedLM): Uses LanguageModelLogitLens
    - **Classification Models** (AutoModelForSequenceClassification): Uses ClassificationLogitLens
    Args:
        model: The wrapped model to analyze
        tokenizer: The tokenizer corresponding to the model
        **kwargs: Additional arguments passed to the specific LogitLens implementation

    Returns:
        Appropriate LogitLens implementation (LanguageModelLogitLens or ClassificationLogitLens)

    Example usage:
        # Create a LogitLens instance for language models
        logit_lens = LogitLens(splitted_model, tokenizer,
                              normalization=True,
                              nb_token=6,
                              batch_size=2)

        # Analyze how predictions evolve across layers
        logit_lens.lens("The cat sat on the")

        # For classification models
        cls_lens = LogitLens(classification_model, tokenizer,
                           pooling_strategy="cls")
        cls_lens.lens("This movie is great!")
    """

    # Mapping of model types to their corresponding LogitLens implementations
    _model_type_to_lens = {
        # Language Models
        modeling_auto.AutoModelForCausalLM: LanguageModelLogitLens,
        modeling_auto.AutoModelForMaskedLM: LanguageModelLogitLens,
        # Classification Models
        modeling_auto.AutoModelForSequenceClassification: ClassificationLogitLens,
        modeling_auto.AutoModelForTokenClassification: ClassificationLogitLens,
    }

    def __new__(cls, model: ModelWithSplitPoints, tokenizer: PreTrainedTokenizer, **kwargs) -> BaseLogitLens:
        """
        Create the appropriate LogitLens implementation based on the model type.

        Args:
            model: The wrapped model to analyze
            tokenizer: The tokenizer corresponding to the model
            **kwargs: Additional arguments passed to the specific LogitLens implementation

        Returns:
            Appropriate LogitLens implementation (LanguageModelLogitLens or ClassificationLogitLens)
        """
        # Get the model's autoclass
        model_autoclass = model.model_autoclass

        # Find the appropriate LogitLens implementation
        for autoclass, lens_class in cls._model_type_to_lens.items():
            if model_autoclass == autoclass:
                print(f"Using {lens_class.__name__} for {autoclass.__name__}")
                return lens_class(model, tokenizer, **kwargs)

        # Fallback: try to detect based on model class name
        model_class_name = model._model.__class__.__name__

        if any(name in model_class_name.lower() for name in ["classification", "classifier"]):
            print(f"Detected classification model from class name: {model_class_name}")
            print("Using ClassificationLogitLens")
            return ClassificationLogitLens(model, tokenizer, **kwargs)
        elif any(name in model_class_name.lower() for name in ["causal", "lm", "masked", "language"]):
            print(f"Detected language model from class name: {model_class_name}")
            print("Using LanguageModelLogitLens")
            return LanguageModelLogitLens(model, tokenizer, **kwargs)
        else:
            raise ValueError(
                f"Could not determine appropriate LogitLens implementation for model type: {model_autoclass}. "
                f"Model class: {model_class_name}. "
                f"Supported autoclasses: {list(cls._model_type_to_lens.keys())}"
            )
