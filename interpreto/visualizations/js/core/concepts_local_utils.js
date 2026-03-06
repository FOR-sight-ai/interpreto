(function () {
    /**
     * ConceptsLocalUtils - Shared helpers for local concept visualizations
     */
    window.ConceptsLocalUtils = {
        /**
         * Build ranked concept entries from a score array
         * @param {Array} scores - Array of concept scores
         * @returns {Array} Sorted entries [{ id, rawValue, absValue }]
         */
        buildRankedEntries(scores) {
            const entries = [];
            for (let conceptId = 0; conceptId < scores.length; conceptId++) {
                const rawValue = typeof scores[conceptId] === "number" ? scores[conceptId] : 0;
                const absValue = Math.abs(rawValue);
                if (absValue === 0) {
                    continue;
                }
                entries.push({ id: conceptId, rawValue, absValue });
            }
            entries.sort((a, b) => b.absValue - a.absValue);
            return entries;
        },

        /**
         * Get maximum absolute activation value for a concept
         * @param {number} conceptId - Concept id
         * @param {Array} activations - 2D activation array
         * @returns {number}
         */
        getMaxAbsForConcept(conceptId, activations) {
            let maxValue = 0;
            for (let tokenIndex = 0; tokenIndex < activations.length; tokenIndex++) {
                const row = activations[tokenIndex] || [];
                const rawValue = typeof row[conceptId] === "number" ? row[conceptId] : 0;
                const absValue = Math.abs(rawValue);
                if (absValue > maxValue) {
                    maxValue = absValue;
                }
            }
            return maxValue;
        },

        /**
         * Resolve a concept color from a colormap or fallback
         * @param {number} conceptId - Concept id
         * @param {object} defaultColormap - Optional concept color map
         * @param {string} fallbackColor - Fallback color
         * @returns {string}
         */
        getConceptColor(conceptId, defaultColormap, fallbackColor) {
            const mapped = StyleComputer.getColorFromMap(defaultColormap, conceptId);
            return mapped || fallbackColor;
        },

        /**
         * Render selectable concept labels
         * @param {HTMLElement} wrapper - Wrapper element for show/hide
         * @param {HTMLElement} container - Container for concept labels
         * @param {Array} topConcepts - Array of concept metadata
         * @param {object} callbacks - { onClick, onMouseOver, onMouseOut }
         * @returns {HTMLElement[]} Rendered concept elements
         */
        renderConceptList(wrapper, container, topConcepts, callbacks = {}) {
            if (!wrapper || !container) {
                return [];
            }

            container.innerHTML = "";

            if (!topConcepts.length) {
                wrapper.classList.add("is-hidden");
                return [];
            }

            wrapper.classList.remove("is-hidden");

            const concepts = topConcepts.map((concept) => ({
                label: concept.label,
                id: concept.id,
                color: concept.color,
            }));
            const { conceptElements } = DOMRenderer.renderConcepts(container, concepts);

            for (let i = 0; i < conceptElements.length; i++) {
                const element = conceptElements[i];
                element.classList.add("reactive-word-style");
                element.dataset.conceptIndex = i.toString();
                if (callbacks.onClick) {
                    element.addEventListener("click", (e) => {
                        e.preventDefault();
                        callbacks.onClick(i);
                    });
                }
                if (callbacks.onMouseOver) {
                    element.addEventListener("mouseover", () => callbacks.onMouseOver(i));
                }
                if (callbacks.onMouseOut) {
                    element.addEventListener("mouseout", () => callbacks.onMouseOut(i));
                }
            }

            return conceptElements;
        },

        /**
         * Clear concept-driven styles for token elements
         * @param {HTMLElement[]} elements
         */
        clearTokenStyles(elements) {
            for (const element of elements) {
                element.style = "";
                DOMRenderer.setTooltip(element, null);
            }
        },

        /**
         * Update tokens with dominant concept highlight
         * @param {HTMLElement[]} elements
         * @param {Array} activations - 2D activation array
         * @param {Array} topConcepts - Ranked concept metadata
         * @param {number[]} backgroundColor - Background RGB
         */
        updateTokensDefault(elements, activations, topConcepts, backgroundColor) {
            for (let tokenIndex = 0; tokenIndex < elements.length; tokenIndex++) {
                const element = elements[tokenIndex];
                const row = activations[tokenIndex] || [];

                let bestConceptIndex = null;
                let bestValue = 0;
                let bestRawValue = 0;

                for (let i = 0; i < topConcepts.length; i++) {
                    const concept = topConcepts[i];
                    const rawValue = typeof row[concept.id] === "number" ? row[concept.id] : 0;
                    const absValue = Math.abs(rawValue);
                    if (absValue > bestValue) {
                        bestValue = absValue;
                        bestRawValue = rawValue;
                        bestConceptIndex = i;
                    }
                }

                if (bestConceptIndex === null || bestValue === 0) {
                    element.style = "";
                    DOMRenderer.setTooltip(element, null);
                    continue;
                }

                const concept = topConcepts[bestConceptIndex];
                const style = StyleComputer.computeConceptStyle(
                    bestValue,
                    concept.maxAbs,
                    concept.color,
                    backgroundColor
                );
                element.style = style;
                DOMRenderer.setTooltip(
                    element,
                    StyleComputer.formatTooltip(bestRawValue)
                );
            }
        },

        /**
         * Update tokens with a single concept highlight
         * @param {HTMLElement[]} elements
         * @param {Array} activations - 2D activation array
         * @param {object} concept - Selected concept metadata
         * @param {number[]} backgroundColor - Background RGB
         */
        updateTokensForConcept(elements, activations, concept, backgroundColor) {
            for (let tokenIndex = 0; tokenIndex < elements.length; tokenIndex++) {
                const element = elements[tokenIndex];
                const row = activations[tokenIndex] || [];
                const rawValue = typeof row[concept.id] === "number" ? row[concept.id] : 0;
                const absValue = Math.abs(rawValue);

                if (absValue === 0) {
                    element.style = "";
                    DOMRenderer.setTooltip(element, null);
                    continue;
                }

                const style = StyleComputer.computeConceptStyle(
                    absValue,
                    concept.maxAbs,
                    concept.color,
                    backgroundColor
                );
                element.style = style;
                DOMRenderer.setTooltip(element, StyleComputer.formatTooltip(rawValue));
            }
        },
    };
})();
