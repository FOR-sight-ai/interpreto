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

# # MIT License
# #
# # Copyright (c) 2025 IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# # rights reserved. DEEL and FOR are research programs operated by IVADO, IRT Saint Exupéry,
# # CRIAQ and ANITI - https://www.deel.ai/.
# #
# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:
# #
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.

# from itertools import product

# import torch
# from nnsight import NNsight
# from tests.fixtures.model_zoo import (
#     # SimpleTokenizer,
#     SmallConv1DModel,
#     SmallConv2DModel,
#     SmallDenseModel,
#     SmallGRUModel,
#     SmallLSTMModel,
#     SmallRNNModel,
#     # SmallTextClassifier,
#     # SmallTextGenerator,
#     SmallViTModel,
# )

# from interpreto.attributions import (
#     IntegratedGradients,
#     Sobol,
# )

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# attribution_methods_to_test = [IntegratedGradients, Sobol]
# attribution_method_args = {
#     IntegratedGradients: {"n_perturbations": 3, "baseline": 0.0},
#     Sobol: {"n_token_perturbations": 3},
# }


# def test_attribution_methods_with_1d_input_models():
#     input_shape = (4, 10)  # (batch_size=4, input_size=10)
#     model = NNsight(SmallDenseModel(input_size=input_shape[1]))
#     input_tensor = torch.randn(input_shape)

#     for attribution_explainer in attribution_methods_to_test:
#         attributions = attribution_explainer(model, batch_size=3, device=DEVICE).explain(input_tensor)

#         assert attributions.shape == input_shape


# def test_attribution_methods_with_2d_input_models():
#     input_shape = (4, 3, 10)  # (batch_size=4, channels=3, input_size=10)
#     models = [
#         NNsight(SmallRNNModel(input_size=input_shape[2])),
#         NNsight(SmallLSTMModel(input_size=input_shape[2])),
#         NNsight(SmallGRUModel(input_size=input_shape[2])),
#         NNsight(SmallConv1DModel(input_channels=input_shape[1], input_length=input_shape[2])),
#     ]

#     input_tensor = torch.randn(input_shape)

#     for model, attribution_explainer in product(models, attribution_methods_to_test):
#         attributions = attribution_explainer(model, batch_size=3, device=DEVICE).explain(input_tensor)
#         assert attributions.shape == input_shape


# def test_attribution_methods_with_3d_input_models():
#     input_shape = (4, 3, 10, 5)  # (batch_size=4, channels=3, input_h=10, input_w=5)
#     models = [
#         NNsight(SmallConv2DModel(input_channels=input_shape[1], image_size=(input_shape[2], input_shape[3]))),
#         NNsight(SmallViTModel(input_channels=input_shape[1], image_size=(input_shape[2], input_shape[3]))),
#     ]

#     input_tensor = torch.randn(input_shape)

#     for model, attribution_explainer in product(models, attribution_methods_to_test):
#         attributions = attribution_explainer(model, batch_size=3, device=DEVICE).explain(input_tensor)

#         assert attributions.shape == input_shape
