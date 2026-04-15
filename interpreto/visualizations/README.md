# explanations/

Goal
- Provide lightweight, notebook-friendly explanation visualizations (attributions + concepts) with a shared look-and-feel and a small, reusable JS/CSS core.

What lives where
- `attributions.py`: Python entrypoints for attribution visualizations (classification + generation). Builds the HTML, loads JS/CSS, passes data + colors.
- `concepts.py`: Python entrypoints for concept visualizations (global classification + local generation). Same HTML/JS pipeline.
- `css/visualization.css`: Shared CSS for all visualizations (fonts, colors, layout helpers).
- `js/core/dom_renderer.js`: Pure DOM rendering helpers (classes, inputs, outputs, concepts, tooltips).
- `js/core/state_manager.js`: Selection/hover state machine.
- `js/core/style_computer.js`: Styling helpers (color rules, label styles, tooltip formatting).
- `js/core/view_updater.js`: Applies styles to DOM elements based on state.
- `js/visualizations/attribution_classification.js`: Attribution visualization for classification.
- `js/visualizations/attribution_generation.js`: Attribution visualization for generation.
- `js/visualizations/concepts_classification_global.js`: Global concept visualization for classification.
- `js/visualizations/concepts_generation_local.js`: Local concept visualization for generation.
- `js/visualizations/concepts_classification_local.js`: Local concept visualization for classification.
- `test.ipynb`: Scratch notebook; ignore when editing production logic.

Interaction rules (must hold everywhere)
- Single selection only: only one element (class/concept/output) can be selected at a time.
- Hover never overrides a selection: when something is selected, hover shows value only and does not change colors.
- Clicking a different element deselects the previous one and selects the new one.

Harmonization rules (must hold everywhere)
- Shared visual language across concepts/attributions/classification/generation.
- Font family and sizes are centralized in `css/visualization.css` via the `--visualization-font-*` variables.
- Default selectable colors come from tab10 (and extended palette when needed).
- When an item is selected, highlight uses a border-only style; text highlight colors come from the `onclick_colormap` (blue/red by default).
- Colors are configurable via Python:
  - `default_colormap`: `{id: color}` overrides for classes/concepts.
  - `onclick_colormap`: `(selected_color, hover_color)` for active label borders.

Behavioral rules
- The JS/CSS should remain factorized: prefer shared helpers in `js/core/style_computer.js`, `js/core/view_updater.js`, and `js/core/dom_renderer.js`.
- Keep the data contract stable across Python and JS (no ad-hoc fields without shared usage).
- Tooltips display values only (no labels mixed into the tooltip content).

Editing notes
- Prefer ASCII-only edits unless a file already uses non-ASCII.
- Keep `test.ipynb` untouched unless explicitly requested.
