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

# Logit lens is not a built-in method from nnsight, but if you want to practice it 
# directly without using our pipeline, you can use the following tutorial:
# https://nnsight.net/notebooks/tutorials/logit_lens/


"""
Implementation of the Logit Lens method for model interpretability.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizer

import matplotlib
matplotlib.rcParams['font.family'] = "Noto Serif"
import matplotlib.pyplot as plt
from interpreto.model_wrapping.model_with_split_points import ModelWithSplitPoints
import torch.nn as nn
import torch

from IPython.display import display, HTML
import matplotlib
import numpy as np

from transformers import BatchEncoding
import torch.nn.functional as F
import numpy as np

def _prob_to_color(prob, min_prob=0.0, max_prob=1.0):
    prob = float((prob - min_prob) / (max_prob - min_prob))
    r = int(255 * prob)
    g = 0
    b = int(255 * (1 - prob))
    return f'rgb({r},{g},{b})'

def _escape_html(text):
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;'))

def _prettify_token(token: str) -> str:
    return token.replace("Ġ", " ").replace("▁", " ").replace("</w>", "")


class LogitLens():
    """
    Logit Lens is a method for interpreting model predictions by analyzing the logits of a model for any given activation.
    It applies the linear function of the logits to the activations of the model, allowing for a deeper understanding of how different parts of the model contribute to the final prediction.

    **Reference:**
    https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

    Examples:
        >>> from interpreto import LogitLens
        >>> method = LogitLens(model=model, tokenizer=tokenizer)
        >>> method.explain(text)
        >>> method.lens(text) # visualizing the logit lens predictions
    """

    def __init__(self, model: ModelWithSplitPoints,
                 tokenizer: PreTrainedTokenizer,
                 head_name: str | None = None,
                 normalization: bool = False,
                 nb_token: int = 0,
                 normalization_method: nn.Module | None = nn.LayerNorm,
                 batch_size: int = 8):
        self.splitted_model = model
        self.model = model._model
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size if tokenizer else None
        self.model_head = None
        self.activations = None
        self.head_name = head_name
        self.normalization = normalization
        self.nb_token = min(nb_token, self.vocab_size)
        self.normalization_method = normalization_method
        self.device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layer_names = None
        self.batch_size = batch_size

        possible_heads = ['lm_head', 'cls', 'score', 'predictions', 'decoder']
        k = 0
        if self.head_name is None:
            print("No head name specified, trying to find a suitable head in the model.")
            for head_name in possible_heads:
                if hasattr(self.model, head_name):
                    head = getattr(self.model, head_name)
                    if isinstance(head, nn.Module):
                        self.model_head = head
                        self.head_name = head_name
                        k += 1
                    elif isinstance(head, object):
                        sub_heads = [attr for attr in dir(head) if not attr.startswith("_")]
                        for sub_head_name in sub_heads:
                            sub_head = getattr(head, sub_head_name)
                            if isinstance(sub_head, nn.Linear):
                                self.model_head = sub_head
                                self.head_name = f"{head_name}.{sub_head_name}"
                                k += 1
        else:
            if hasattr(self.model, self.head_name):
                self.model_head = getattr(self.model, self.head_name)
                k += 1
                if not isinstance(self.model_head, nn.Module):
                    raise ValueError(f"The pre-set head '{self.head_name}' is not a valid nn.Module.")
            else:
                raise ValueError(f"The specified head '{self.head_name}' does not exist in the model.")
        if k == 0:
            raise ValueError("No known classifier head found in the model. Please specify a valid head name or ensure the model has a compatible linear head.")
        elif k > 1:
            raise ValueError(f"Multiple classifier heads found in the model. Please specify a valid head name. Found: {k} possible heads.")
        self.features_dim = self.model_head.in_features if self.model_head else None

        
    def do_lens(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Apply the Logit Lens method to the model activations.
        This method computes the logits for the given activations using the model head.
        """
        if not self.model_head:
            raise ValueError("Model head is not set. Please make sure the initialization succeded before applying the lens.")

        if self.normalization:
            activation = self.normalization_method(activation.size(-1))(activation)
        activation = activation.to(self.device)
        logits = self.model_head(activation)
        proba = F.softmax(logits, dim=-1)
        proba = proba.detach().cpu()
        return proba
    
    def _explain(self,
                inputs: str | list[str] | BatchEncoding | torch.Tensor,
                layers_name: str | list[str] | None = None):
        """
        Generate explanations using the Logit Lens method.
        This method applies the linear function of the logits to the activations of the model.
        """
        if isinstance(inputs, (str, list)): 
            inputs_model = self.tokenizer(inputs, 
                                        return_tensors='pt', 
                                        padding=True, 
                                        truncation=True)
            model_input_keys = self.model.forward.__code__.co_varnames
            for key in list(inputs_model.keys()):
                if key not in model_input_keys:
                    inputs_model.pop(key)
        else :
            inputs_model = inputs
        # print(inputs_model["input_ids"].shape)
        mask_idxs = inputs_model["attention_mask"].bool().unsqueeze(-1).expand(-1, -1, self.vocab_size)
        self.activations = self.splitted_model.get_activations(inputs_model, ModelWithSplitPoints.activation_granularities.ALL)

        if layers_name is None:
            layers_name = list(self.activations.keys())

        for layer in layers_name:
            if layer not in self.activations:
                raise ValueError(f"Layer '{layer}' not found in the model activations. Please check the layer names.")

        for layer, activation in self.activations.items():
            if activation.shape[-1] != self.features_dim:
                raise ValueError(f"Activation shape mismatch for layer '{layer}': expected {self.features_dim}, got {activation.shape[-1]}. Please check the model and the layer names.")

        if not self.model_head:
            raise ValueError("Model head is not set. Please call verify the initialization before explaining.")
        
        if isinstance(layers_name, str):
            layers_name = [layers_name]

        self.layer_names = layers_name
        # print(self.activations[self.layer_names[0]].shape)

        proba_dict = {}
        for layer in layers_name:
            proba_dict[layer] = self.do_lens(self.activations[layer])
        if self.nb_token == 0:
            print("WARNING: `nb_token` is set to 0, no top-k tokens will be returned. If you want to see the top-k tokens, please set `nb_token` to a positive integer.")
            return proba_dict
        else:
            top_k_tokens = {}
            for layer, proba in proba_dict.items():
                top_k_indices = torch.topk(proba, self.nb_token, dim=-1).indices
                top_k_tokens[layer] = {}
                top_k_tokens[layer]['tokens'] = np.array([
                    [
                    [self.tokenizer.decode([index.item()]) for index in top_k_indices[batch_idx, seq_idx]]
                    for seq_idx in range(top_k_indices.shape[1])
                    ]
                    for batch_idx in range(top_k_indices.shape[0])
                ])
                top_k_tokens[layer]['proba'] = np.array([
                    [
                    proba[batch_idx, seq_idx, top_k_indices[batch_idx, seq_idx]].detach().cpu().numpy()
                    for seq_idx in range(top_k_indices.shape[1])
                    ]
                    for batch_idx in range(top_k_indices.shape[0])
                ])
            return top_k_tokens
    
    def explain(self,
                inputs: str | list[str] | BatchEncoding | torch.Tensor,
                layers_name: str | list[str] | None = None):
        """
        Generate explanations using the Logit Lens method.
        This part of the method assures that inputs are batched and handles padding mismatches.
        """
        if isinstance(inputs, (str, list)):
            if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id < 0:
                print("Tokenizer does not have a padding token. Setting a default padding token.")
                self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.pad_token})
            if isinstance(inputs, str):
                inputs = [inputs]
            batched_inputs = [inputs[i:i + self.batch_size] for i in range(0, len(inputs), self.batch_size)]
        elif isinstance(inputs, BatchEncoding):
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            batched_inputs = [
                BatchEncoding({
                    "input_ids": input_ids[i:i + self.batch_size],
                    "attention_mask": attention_mask[i:i + self.batch_size]
                })
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
                    max_seq_len = max(merged_results[layer]['tokens'].shape[1], data['tokens'].shape[1])
                    pad_token = self.tokenizer.pad_token
                    pad_token_array = np.array([[pad_token] * max_seq_len])

                    merged_tokens = np.concatenate(
                        (
                            np.pad(merged_results[layer]['tokens'], 
                                   ((0, 0), (0, max_seq_len - merged_results[layer]['tokens'].shape[1]), (0, 0)), 
                                   constant_values=pad_token),
                            np.pad(data['tokens'], 
                                   ((0, 0), (0, max_seq_len - data['tokens'].shape[1]), (0, 0)), 
                                   constant_values=pad_token)
                        ),
                        axis=0
                    )
                    
                    merged_proba = np.concatenate(
                        (
                            np.pad(merged_results[layer]['proba'], 
                                   ((0, 0), (0, max_seq_len - merged_results[layer]['proba'].shape[1]), (0, 0)), 
                                   constant_values=0.0),
                            np.pad(data['proba'], 
                                   ((0, 0), (0, max_seq_len - data['proba'].shape[1]), (0, 0)), 
                                   constant_values=0.0)
                        ),
                        axis=0
                    )

                    merged_results[layer]['tokens'] = merged_tokens
                    merged_results[layer]['proba'] = merged_proba

        return merged_results
    
    def __call__(self,
                inputs: str | list[str] | BatchEncoding | torch.Tensor,
                layers_name: str | list[str] | None = None):
        """
        Generate explanations using the Logit Lens method.
        This method applies the linear function of the logits to the activations of the model.
        """
        return self.explain(inputs, layers_name)

    def visualize_logit_lens_interactive(self, 
                                     inputs: str | list[str] | 'BatchEncoding' | 'torch.Tensor', 
                                     layers_name: str | list[str] | None = None):
        """
        Interactive HTML/JS visualization for Logit Lens predictions.
        Each input token is shown as plain text (tokenization artifacts prettified).
        On mouseover, a tooltip shows the top-k predicted tokens and their probabilities
        (probabilities are visible and color-coded).
        """
        explaining = self.explain(inputs, layers_name)
        layers_name = self.layer_names if layers_name is None else layers_name

        if self.nb_token == 0:
            print("WARNING: `nb_token` is set to 0, no top-k tokens will be returned. If you want to see the top-k tokens, please set `nb_token` to a positive integer.")
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
            topk_tokens = layer_data['tokens']
            topk_proba = layer_data['proba']
            for batch_idx, sentence in enumerate(tokens if nb_inputs > 1 else [tokens]):
                html += "<div style='margin-bottom:12px;'>"
                for tok_idx, token in enumerate(sentence):
                    if tok_idx >= topk_tokens.shape[1]: 
                        continue
                    tk_toks = topk_tokens[batch_idx, tok_idx]
                    tk_probs = topk_proba[batch_idx, tok_idx]
                    tooltip_html = "<div><b>Top-k predictions:</b><ul style='padding-left:16px;margin:0;'>"
                    for tk, prob in zip(tk_toks, tk_probs):
                        color = _prob_to_color(prob)
                        tk_esc = _escape_html(_prettify_token(str(tk)))
                        tooltip_html += f"<li><span style='color:{color};font-weight:bold;'>{tk_esc}</span>: {prob:.3f}</li>"
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
    
    def lens(self,
            inputs: str | list[str] | BatchEncoding | torch.Tensor,
            layers_name: str | list[str] | None = None):
        """
        Plot a readable lens
        """
        
        if isinstance(layers_name, str):
            layers_name = [layers_name]

        if isinstance(inputs, list):
            n_inputs = len(inputs)
        elif isinstance(inputs, str):
            n_inputs = 1
        elif isinstance(inputs, BatchEncoding):
            n_inputs = inputs.input_ids.shape[0]
        else:
            n_inputs = inputs.shape[0] if isinstance(inputs, torch.Tensor) else 1

        if isinstance(inputs, torch.Tensor) and len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(0)
            n_inputs = 1
        elif isinstance(inputs, torch.Tensor) and len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(0).unsqueeze(0)
            n_inputs = 1
        
        explaining = self.explain(inputs, layers_name)
        layers_name = self.layer_names if layers_name is None else layers_name

        if self.nb_token == 0:
            print("WARNING: `nb_token` is set to 0, no top-k tokens will be returned. If you want to see the top-k tokens, please set `nb_token` to a positive integer.")
            return explaining
        
        self.visualize_logit_lens_interactive(inputs, layers_name)