(function () {
    /**
     * DOMRenderer - Creates DOM elements for visualizations
     */
    window.DOMRenderer = {
        /**
         * Normalize special characters in a word for display
         * @param {string} word - The word to normalize
         * @returns {string} The normalized word
         */
        normalizeSpecialChars(word) {
            return word
                .replace(/\n/g, "\\n")
                .replace(/\r/g, "\\r")
                .replace(/\t/g, "\\t");
        },

        /**
         * Create a single class label element (for single-class display)
         * @param {HTMLElement} container - The container element
         * @param {object} classData - The class data { name }
         * @returns {HTMLElement} The created element
         */
        renderClassLabel(container, classData) {
            const element = document.createElement("div");
            element.classList.add("common-word-style", "class-style");
            element.textContent = classData.name;
            element.dataset.classId = "0";
            container.appendChild(element);
            return element;
        },

        /**
         * Create class button elements (for multi-class display)
         * @param {HTMLElement} container - The container element
         * @param {Array} classesData - Array of class data objects
         * @param {object} callbacks - Event callbacks { onClick, onMouseOver, onMouseOut }
         * @returns {HTMLElement[]} Array of created elements
         */
        renderClassButtons(container, classesData, callbacks, options = {}) {
            const { useColors = false } = options;
            const elements = [];

            for (let i = 0; i < classesData.length; i++) {
                const classData = classesData[i];
                const element = document.createElement("button");
                element.classList.add(
                    "common-word-style",
                    "highlighted-word-style",
                    "reactive-word-style",
                    "class-style"
                );
                element.textContent = classData.name;
                element.dataset.classId = i.toString();
                element.type = "button"; // Ensure it's not a submit button
                if (useColors && classData.color) {
                    element.dataset.classColor = classData.color;
                }

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

                container.appendChild(element);
                elements.push(element);
            }

            return elements;
        },

        /**
         * Create class button elements with color indicators (for multi-class display)
         * @param {HTMLElement} container - The container element
         * @param {Array} classesData - Array of class data objects
         * @param {object} callbacks - Event callbacks { onClick, onMouseOver, onMouseOut }
         * @returns {HTMLElement[]} Array of created elements
         */
        renderClassButtonsWithColors(container, classesData, callbacks) {
            return this.renderClassButtons(container, classesData, callbacks, { useColors: true });
        },

        /**
         * Create input word elements
         * @param {HTMLElement} container - The container element
         * @param {Array} words - Array of words
         * @returns {object} { sentenceElement, wordElements }
         */
        renderInputs(container, words) {
            const sentenceElement = document.createElement("div");
            sentenceElement.classList.add("line-style");

            const wordElements = [];
            for (let j = 0; j < words.length; j++) {
                const word = this.normalizeSpecialChars(words[j]);
                const wordElement = document.createElement("div");
                wordElement.classList.add("common-word-style", "highlighted-word-style");
                wordElement.textContent = word;
                wordElement.dataset.wordIndex = j.toString();
                sentenceElement.appendChild(wordElement);
                wordElements.push(wordElement);
            }

            container.appendChild(sentenceElement);
            return { sentenceElement, wordElements };
        },

        /**
         * Create output word button elements
         * @param {HTMLElement} container - The container element
         * @param {Array} words - Array of words
         * @param {object} callbacks - Event callbacks { onClick, onMouseOver, onMouseOut }
         * @returns {HTMLElement[]} Array of created elements
         */
        renderOutputs(container, words, callbacks) {
            const elements = [];

            for (let i = 0; i < words.length; i++) {
                const word = this.normalizeSpecialChars(words[i]);
                const element = document.createElement("button");
                element.classList.add(
                    "common-word-style",
                    "highlighted-word-style",
                    "reactive-word-style"
                );
                element.textContent = word;
                element.dataset.outputIndex = i.toString();

                if (callbacks.onClick) {
                    element.onclick = () => callbacks.onClick(i);
                }
                if (callbacks.onMouseOver) {
                    element.onmouseover = () => callbacks.onMouseOver(i);
                }
                if (callbacks.onMouseOut) {
                    element.onmouseout = () => callbacks.onMouseOut(i);
                }

                container.appendChild(element);
                elements.push(element);
            }

            return elements;
        },

        /**
         * Create concept label elements
         * @param {HTMLElement} container - The container element
         * @param {Array} concepts - Array of concept objects { label }
         * @returns {object} { lineElement, conceptElements }
         */
        renderConcepts(container, concepts) {
            const lineElement = document.createElement("div");
            lineElement.classList.add("line-style");

            const conceptElements = [];
            for (let i = 0; i < concepts.length; i++) {
                const concept = concepts[i];
                const label = Array.isArray(concept.label)
                    ? concept.label.join("\n")
                    : String(concept.label);

                const element = document.createElement("div");
                element.classList.add(
                    "common-word-style",
                    "highlighted-word-style",
                    "concept-style"
                );
                element.textContent = label;
                if (concept.id !== undefined && concept.id !== null) {
                    element.dataset.conceptId = concept.id.toString();
                }
                if (concept.color) {
                    element.dataset.conceptColor = concept.color;
                }
                element.dataset.conceptIndex = i.toString();
                lineElement.appendChild(element);
                conceptElements.push(element);
            }

            container.appendChild(lineElement);
            return { lineElement, conceptElements };
        },

        /**
         * Create or update a tooltip on an element
         * @param {HTMLElement} element - The element to attach tooltip to
         * @param {string|null} text - Tooltip text, or null to remove
         */
        setTooltip(element, text) {
            // Remove existing tooltip
            const existing = element.querySelector(".tooltiptext");
            if (existing) {
                existing.remove();
            }

            // Add new tooltip if text provided
            if (text !== null) {
                const tooltip = document.createElement("span");
                tooltip.classList.add("tooltiptext");
                tooltip.textContent = text;
                element.appendChild(tooltip);
            }
        },
    };
})();
