(function () {
    /** Query-first retrieval view. Display scores and image assets are prepared in Python. */
    window.ImageRetrievalVisualization = class ImageRetrievalVisualization {
        constructor(rootId, data) {
            // A running notebook can retain the previous Python function while
            // the HTML builder loads this newer JS from disk on every display.
            // Adapt that payload before rendering so it cannot leave a partial view.
            if (data.token_highlights === undefined) {
                const magnitudes = data.token_scores.map(Math.abs);
                const range = Math.max(...magnitudes);
                const hex = (rgb) => "#" + rgb.map((value) => value.toString(16).padStart(2, "0")).join("");
                data = {
                    ...data,
                    token_highlights: data.token_scores.map((value) =>
                        range === 0 ? 0 : value / range
                    ),
                    token_style: {
                        positive_color: hex(data.colors[0]),
                        negative_color: hex(data.colors[1]),
                        min: -1,
                        max: 1,
                    },
                    aggregate: data.aggregate.src,
                    maps: data.maps.map((map) => map.src),
                };
            }
            this.root = document.getElementById(rootId);
            this.data = data;
            this.selected = null;
            this.hovered = null;
            this.focused = null;
            const add = (tag, parent, text, className) => {
                const element = document.createElement(tag);
                if (text !== undefined) element.textContent = text;
                if (className) element.className = className;
                parent.appendChild(element);
                return element;
            };
            add("h3", this.root, "Query");
            const query = add("div", this.root, undefined, "line-style");
            this.buttons = DOMRenderer.renderOutputs(query, data.tokens, {});
            this.buttons.forEach((button, index) => {
                button.classList.add("retrieval-token");
                button.type = "button";
                button.style.cssText = StyleComputer.computeWordStyle(data.token_highlights[index], data.token_style);
                DOMRenderer.setTooltip(button, StyleComputer.formatTooltip(data.token_scores[index]));
                button.addEventListener("mouseenter", () => { this.hovered = index; this.refresh(); });
                button.addEventListener("mouseleave", () => { this.hovered = null; this.refresh(); });
                // Pointer focus must not linger as a preview after mouseleave.
                button.addEventListener("focus", () => {
                    if (button.matches(":focus-visible")) { this.focused = index; this.refresh(); }
                });
                button.addEventListener("blur", () => { this.focused = null; this.refresh(); });
                button.addEventListener("click", () => {
                    this.selected = this.selected === index ? null : index;
                    this.hovered = null;
                    this.focused = null;
                    this.refresh();
                });
            });
            this.root.addEventListener("keydown", (event) => {
                if (event.key === "Escape") { event.preventDefault(); this.clear(); }
            });
            add("h3", this.root, "Image");
            const stage = add("div", this.root, undefined, "retrieval-image");
            const image = add("img", stage);
            image.src = data.image;
            image.alt = "Retrieved image at model input resolution";
            this.overlay = add("img", stage, undefined, "retrieval-overlay");
            this.overlay.alt = "";
            this.refresh();
        }

        clear() {
            this.selected = this.hovered = this.focused = null;
            this.refresh();
        }

        refresh() {
            const active = this.hovered ?? this.focused ?? this.selected;
            const map = active === null ? this.data.aggregate : this.data.maps[active];
            this.overlay.src = map;
            this.buttons.forEach((button, index) => {
                button.setAttribute("aria-pressed", String(index === this.selected));
                button.classList.toggle("selected-style", index === this.selected);
                button.classList.toggle("is-emphasized", index === active);
            });
        }
    };
})();
