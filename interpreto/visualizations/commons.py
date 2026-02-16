import os


def _load_js_files() -> str:
    """Load all JS files required for visualizations."""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    js_files = [
        os.path.join("js", "core", "state_manager.js"),
        os.path.join("js", "core", "style_computer.js"),
        os.path.join("js", "core", "dom_renderer.js"),
        os.path.join("js", "core", "view_updater.js"),
        os.path.join("js", "visualizations", "concepts_classification_global.js"),
        os.path.join("js", "visualizations", "concepts_generation_local.js"),
        os.path.join("js", "visualizations", "concepts_classification_local.js"),
        os.path.join("js", "visualizations", "attribution_classification.js"),
        os.path.join("js", "visualizations", "attribution_generation.js"),
    ]

    js_content = ""
    for js_file in js_files:
        js_file_path = os.path.join(current_dir, js_file)
        with open(js_file_path, encoding="utf-8") as file:
            js_content += file.read() + "\n"

    return js_content


def _build_html_header(custom_css: str) -> str:
    """Build the HTML header with JS and CSS content."""
    # Load the JS and CSS files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    js_content = _load_js_files()

    css_file_path = os.path.join(current_dir, "css", "visualization.css")
    with open(css_file_path, encoding="utf-8") as file:
        css = file.read()

    style_block = css if not custom_css else f"{css}\n{custom_css}"
    return (
        "<head>"
        "<style>\n"
        f"{style_block}\n"
        "</style>"
        "<script>\n"
        f"{js_content}\n"
        "</script>"
        "</head>"
        '<body class="body-visualization">'
    )


def _generate_distinct_colors(n: int) -> list[str]:
    """Generate n visually distinct colors, using tab10 as the default palette."""
    tab10 = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    if n <= len(tab10):
        return tab10[:n]

    colors = list(tab10)
    for i in range(len(tab10), n):
        # Distribute hues evenly around the color wheel
        hue = (i * 360 / n) % 360
        # Use high saturation and lower lightness for vivid colors with good contrast
        saturation = 80
        lightness = 40  # Darker for better white text contrast

        # Convert HSL to RGB
        h = hue / 360
        s = saturation / 100
        l = lightness / 100

        if s == 0:
            r = g = b = l
        else:

            def hue_to_rgb(p, q, t):
                if t < 0:
                    t += 1
                if t > 1:
                    t -= 1
                if t < 1 / 6:
                    return p + (q - p) * 6 * t
                if t < 1 / 2:
                    return q
                if t < 2 / 3:
                    return p + (q - p) * (2 / 3 - t) * 6
                return p

            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue_to_rgb(p, q, h + 1 / 3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1 / 3)

        # Convert RGB to hex
        colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return colors


def _normalize_colormap(
    default_colormap: dict[int, str] | None,
    *,
    name: str = "default_colormap",
) -> dict[int, str]:
    if default_colormap is None:
        return {}
    if not isinstance(default_colormap, dict):
        raise TypeError(f"{name} must be a dict[int, str] or None.")
    normalized: dict[int, str] = {}
    for key, value in default_colormap.items():
        if value is None:
            continue
        try:
            key_int = int(key)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} keys must be int-like.") from exc
        normalized[key_int] = str(value)
    return normalized


def _normalize_onclick_colormap(
    onclick_colormap: tuple[str, str] | list[str] | None,
) -> tuple[str, str]:
    if onclick_colormap is None:
        return ("#ff0000", "#0000ff")
    if isinstance(onclick_colormap, (list, tuple)) and len(onclick_colormap) == 2:
        return (str(onclick_colormap[0]), str(onclick_colormap[1]))
    raise TypeError("onclick_colormap must be a tuple or list of two strings.")


def _build_default_colormap(
    ids: list[int],
    default_colormap: dict[int, str] | None,
) -> dict[int, str]:
    unique_ids: list[int] = []
    seen: set[int] = set()
    for id_value in ids:
        if id_value in seen:
            continue
        seen.add(id_value)
        unique_ids.append(id_value)

    colors = _generate_distinct_colors(len(unique_ids))
    colormap = {id_value: colors[index] for index, id_value in enumerate(unique_ids)}
    overrides = _normalize_colormap(default_colormap)
    for key, value in overrides.items():
        if key in colormap:
            colormap[key] = value
    return colormap


def _save_html(html: str, save_path: str | os.PathLike[str]) -> None:
    """Save the rendered HTML to disk."""
    path = os.fspath(save_path)
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(html)
