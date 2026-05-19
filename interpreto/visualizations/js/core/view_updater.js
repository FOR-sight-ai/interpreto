(function () {
    /**
     * ViewUpdater - Applies styles and tooltips to DOM elements based on state
     */
    window.ViewUpdater = {
        /**
         * Update class elements styles
         * @param {HTMLElement[]} elements - Array of class elements
         * @param {object} state - Current state { activatedClassId }
         */
        updateClasses(elements, state, options = {}) {
            const {
                showClassColorsWhenInactive = true,
                highlightActiveText = false,
                useSelectedStyle = true,
                onClickColorMap = null,
            } = options;
            const hasActiveClass = state.activatedClassId !== null;
            const showDefaultBackground = showClassColorsWhenInactive && !hasActiveClass;

            for (const element of elements) {
                const classId = parseInt(element.dataset.classId, 10);
                const isActive = classId === state.activatedClassId;
                const isSelected = classId === state.selectedClassId;
                const classColor = showClassColorsWhenInactive
                    ? element.dataset.classColor
                    : null;

                if (useSelectedStyle) {
                    element.classList.toggle("selected-style", isSelected);
                } else {
                    element.classList.remove("selected-style");
                }

                const style = StyleComputer.buildLabelStyle(
                    classColor,
                    isActive,
                    isSelected,
                    {
                        onClickColorMap,
                        enableHighlight: highlightActiveText,
                        showDefaultBackground,
                    }
                );
                element.style.cssText = style;
                element.classList.toggle("is-emphasized", isActive || isSelected);
            }
        },

        /**
         * Update input word elements styles
         * @param {HTMLElement[]} wordElements - Array of word elements
         * @param {object} state - Current state { activatedClassId, currentOutputId }
         * @param {object} data - Data { inputs, classes, custom_style }
         * @param {object} options - Style options { highlightBorder }
         */
        updateInputs(wordElements, state, data, options = {}) {
            const customStyle = StyleComputer.buildCustomStyle(data.custom_style);
            const { highlightBorder = false } = options;

            for (let j = 0; j < wordElements.length; j++) {
                const wordElement = wordElements[j];

                if (state.activatedClassId === null || state.currentOutputId === null) {
                    // Reset style
                    wordElement.style = customStyle;
                    DOMRenderer.setTooltip(wordElement, null);
                } else {
                    // Compute and apply attribution style
                    const alpha = data.inputs.attributions[state.currentOutputId][j][state.activatedClassId];
                    const classMeta = data.classes[state.activatedClassId];

                    const style = StyleComputer.computeWordStyle(alpha, classMeta, {
                        normalize: true,
                        highlightBorder,
                    });
                    wordElement.style = style + customStyle;

                    // Update tooltip
                    DOMRenderer.setTooltip(wordElement, StyleComputer.formatTooltip(alpha));
                }
            }
        },

        /**
         * Update input word elements styles for default multi-class view
         * Each token is colored by its dominant class (highest positive attribution)
         * @param {HTMLElement[]} wordElements - Array of word elements
         * @param {object} state - Current state { currentOutputId }
         * @param {object} data - Data { inputs, classes, custom_style }
         * @param {object} options - Style options { highlightBorder }
         */
        updateInputsDefaultMultiClass(wordElements, state, data, options = {}) {
            const customStyle = StyleComputer.buildCustomStyle(data.custom_style);
            const { highlightBorder = false } = options;

            // Compute global max for normalization across all classes
            let globalMax = 0;
            for (let j = 0; j < wordElements.length; j++) {
                for (let c = 0; c < data.classes.length; c++) {
                    const val = data.inputs.attributions[state.currentOutputId][j][c];
                    if (val > globalMax) {
                        globalMax = val;
                    }
                }
            }

            for (let j = 0; j < wordElements.length; j++) {
                const wordElement = wordElements[j];

                // Get attributions for all classes for this token
                const attributionsPerClass = data.inputs.attributions[state.currentOutputId][j];

                // Find dominant class (highest positive attribution)
                const { classId: dominantClassId, value: dominantValue } =
                    StyleComputer.findDominantClass(attributionsPerClass);

                if (dominantClassId === null) {
                    // No positive attributions - reset style
                    wordElement.style = customStyle;
                    DOMRenderer.setTooltip(wordElement, null);
                } else {
                    // Apply color of dominant class
                    const classMeta = data.classes[dominantClassId];
                    const style = StyleComputer.computeDefaultClassStyle(
                        dominantValue,
                        classMeta.color,
                        globalMax,
                        { highlightBorder }
                    );
                    wordElement.style = style + customStyle;

                    // Show tooltip with value only
                    DOMRenderer.setTooltip(
                        wordElement,
                        StyleComputer.formatTooltip(dominantValue)
                    );
                }
            }
        },

        /**
         * Update output word elements styles
         * @param {HTMLElement[]} elements - Array of output elements
         * @param {object} state - Current state { activatedClassId, currentOutputId }
         * @param {object} data - Data { outputs, classes, custom_style }
         * @param {object} options - Style options { highlightBorder }
         */
        updateOutputs(elements, state, data, options = {}) {
            const customStyle = StyleComputer.buildCustomStyle(data.custom_style);
            const { highlightBorder = false } = options;

            for (let i = 0; i < elements.length; i++) {
                const element = elements[i];

                // Update CSS classes for position-based styling
                const isBeforeCurrent = state.currentOutputId !== null && i < state.currentOutputId;
                const isCurrent = i === state.currentOutputId;

                element.classList.toggle("highlighted-word-style", isBeforeCurrent);
                element.classList.toggle("selected-style", isCurrent);

                if (state.activatedClassId !== null && isBeforeCurrent && data.outputs.attributions) {
                    // Compute and apply attribution style
                    const alpha = data.outputs.attributions[state.currentOutputId][i][state.activatedClassId];
                    const classMeta = data.classes[state.activatedClassId];

                    const style = StyleComputer.computeWordStyle(alpha, classMeta, {
                        normalize: true,
                        highlightBorder,
                    });
                    element.style = style + customStyle;

                    // Update tooltip
                    DOMRenderer.setTooltip(element, StyleComputer.formatTooltip(alpha));
                } else {
                    // Reset style
                    element.style = customStyle;
                    DOMRenderer.setTooltip(element, null);
                }
            }
        },
    };
})();
