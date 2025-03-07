"""
Helper functions for generating colormaps for concepts.
"""

from __future__ import annotations

from typing import List

import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt


def make_random_colors(nb_concepts: int) -> List[List[float]]:
    """
    Generate random colors for the concepts.

    Args:
        nb_concepts (int): Number of concepts

    Returns:
        List[List[float]]: List of colors in RGB format
    """
    colors = []
    for i in range(nb_concepts):
        hue = i / nb_concepts
        rgb = hsv_to_rgb([hue, 1, 1]).tolist()
        colors.append(rgb)
    # randomize the order
    np.random.shuffle(colors)
    return colors


def display_color_gradients(colors_list: List[dict]):
    """
    Display a gradient for each color in the list.

    Args:
        colors_list (List[dict]): List of colors in RGB format
    """
    plt.figure(figsize=(10, 2))
    for i, color in enumerate(colors_list):
        gradient = np.zeros((1, 256, 4))
        gradient[:, :, :3] = color[:3]  # Set RGB channels
        gradient[:, :, 3] = np.linspace(0, 1, 256)  # Vary alpha channel
        plt.imshow(gradient, aspect="auto", extent=[i, i + 1, 0, 1])

    plt.xlim(0, len(colors_list))
    plt.axis("off")
    plt.show()
