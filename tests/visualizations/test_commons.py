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

# MIT License
#
# Copyright (c) 2025 IRT Antoine de Saint Exupery et Universite Paul Sabatier Toulouse III - All
# rights reserved. DEEL and FOR are research programs operated by IVADO, IRT Saint Exupery,
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

import pytest

from interpreto.visualizations.commons import (
    _build_default_colormap,
    _build_html_header,
    _normalize_colormap,
    _normalize_onclick_colormap,
)


def test_build_html_header_includes_assets_and_custom_css():
    custom_css = ".custom-test { color: red; }"
    header = _build_html_header(custom_css)

    assert "<head>" in header
    assert "<style>" in header
    assert "<script>" in header
    assert "body-visualization" in header
    assert ".body-visualization" in header
    assert "StateManager" in header
    assert custom_css in header


def test_normalize_colormap_casts_keys_and_drops_none():
    result = _normalize_colormap({"1": "red", 2: None, 3.0: "blue"})
    assert result == {1: "red", 3: "blue"}


def test_normalize_onclick_colormap_defaults_and_validates():
    assert _normalize_onclick_colormap(None) == ("#ff0000", "#0000ff")
    assert _normalize_onclick_colormap(["#111111", "#222222"]) == ("#111111", "#222222")
    with pytest.raises(TypeError, match="onclick_colormap must be a tuple or list of two strings"):
        _normalize_onclick_colormap(["#111111"])


def test_build_default_colormap_respects_order_and_overrides():
    colormap = _build_default_colormap([2, 2, 1], {"1": "#abcdef"})
    assert list(colormap.keys()) == [2, 1]
    assert colormap[1] == "#abcdef"
    assert colormap[2].startswith("#")
    assert colormap[2] != colormap[1]
