(function () {
    /**
     * ClassificationLocalConceptsVisualization - Visualization for local concepts in classification tasks
     */
    window.ClassificationLocalConceptsVisualization = class ClassificationLocalConceptsVisualization {
        /**
         * @param {string} uniqueIdClasses - The unique id of the div containing the classes
         * @param {string} uniqueIdConcepts - The unique id of the div containing the concepts
         * @param {string} uniqueIdConceptsWrapper - The unique id of the concepts wrapper div
         * @param {string} uniqueIdSample - The unique id of the div containing the sample
         * @param {string} jsonData - The JSON data containing classes, concepts, and sample
         */
        constructor(
            uniqueIdClasses,
            uniqueIdConcepts,
            uniqueIdConceptsWrapper,
            uniqueIdSample,
            jsonData
        ) {
            console.log("Creating ClassificationLocalConceptsVisualization");

            this.uniqueIdClasses = uniqueIdClasses;
            this.uniqueIdConcepts = uniqueIdConcepts;
            this.uniqueIdConceptsWrapper = uniqueIdConceptsWrapper;
            this.uniqueIdSample = uniqueIdSample;
            this.data = JSON.parse(jsonData);

            this.sample = Array.isArray(this.data.sample) ? this.data.sample : [];
            this.activations = Array.isArray(this.data.activations)
                ? this.data.activations
                : [];
            this.activationsByClass = this.data.activations_by_class || null;
            this.importances = Array.isArray(this.data.importances)
                ? this.data.importances
                : [];
            this.labels = Array.isArray(this.data.labels) ? this.data.labels : [];
            this.labelsByClass = this.data.labels_by_class || null;

            this.topK = Math.max(0, parseInt(this.data.top_k || 0, 10));
            this.conceptColor = this.data.concept_color || "#f39c12";
            this.defaultColormap = this.data.default_colormap || {};
            this.onClickColorMap =
                Array.isArray(this.data.onclick_colormap) &&
                this.data.onclick_colormap.length >= 2
                    ? this.data.onclick_colormap
                    : null;

            this.backgroundColor = StyleComputer.getBackgroundRgb();
            this.classElements = [];
            this.sampleElements = [];
            this.conceptElements = [];
            this.topConcepts = [];

            this.hoveredClassId = null;
            this.selectedClassId = null;
            this.hoveredConceptIndex = null;
            this.selectedConceptIndex = null;

            // Create DOM
            this._createClasses();
            this._createSample();
            this._setTopConcepts(this._computeDefaultTopConcepts());
            this._renderConcepts();

            // Initial render
            this._refreshAll();
        }

        /**
         * Create class elements in the DOM
         */
        _createClasses() {
            const container = document.getElementById(this.uniqueIdClasses);
            if (!container) return;

            this.classElements = DOMRenderer.renderClassButtons(
                container,
                this.data.classes || [],
                {
                    onClick: (id) => this._onClassClick(id),
                    onMouseOver: (id) => this._onClassMouseOver(id),
                    onMouseOut: (id) => this._onClassMouseOut(id),
                },
                { useColors: false }
            );
        }

        /**
         * Create sample elements in the DOM
         */
        _createSample() {
            const container = document.getElementById(this.uniqueIdSample);
            if (!container) return;

            const { wordElements } = DOMRenderer.renderInputs(container, this.sample);
            this.sampleElements = wordElements;
        }

        /**
         * Render concept elements in the DOM
         */
        _renderConcepts() {
            const wrapper = document.getElementById(this.uniqueIdConceptsWrapper);
            const container = document.getElementById(this.uniqueIdConcepts);
            if (!wrapper || !container) return;

            container.innerHTML = "";

            if (!this.topConcepts.length) {
                wrapper.classList.add("is-hidden");
                this.conceptElements = [];
                return;
            }

            wrapper.classList.remove("is-hidden");

            const concepts = this.topConcepts.map((concept) => ({
                label: concept.label,
                id: concept.id,
                color: concept.color,
            }));
            const { conceptElements } = DOMRenderer.renderConcepts(container, concepts);
            this.conceptElements = conceptElements;

            for (let i = 0; i < conceptElements.length; i++) {
                const element = conceptElements[i];
                element.classList.add("reactive-word-style");
                element.dataset.conceptIndex = i.toString();
                element.addEventListener("click", (e) => {
                    e.preventDefault();
                    this._onConceptClick(i);
                });
                element.addEventListener("mouseover", () => this._onConceptMouseOver(i));
                element.addEventListener("mouseout", () => this._onConceptMouseOut(i));
            }
        }

        /**
         * Refresh all view components
         */
        _refreshAll() {
            this._refreshClasses();
            this._refreshConcepts();
            this._refreshTokens();
        }

        /**
         * Handle class click event
         * @param {number} classId
         */
        _onClassClick(classId) {
            const wasSelected = this.selectedClassId === classId;
            this.selectedClassId = wasSelected ? null : classId;
            this.hoveredClassId = null;

            if (wasSelected) {
                this._setTopConcepts(this._computeDefaultTopConcepts());
            } else {
                this._setTopConcepts(this._computeClassTopConcepts(classId));
            }

            this._renderConcepts();
            this._refreshAll();
        }

        /**
         * Handle class mouse over event
         * @param {number} classId
         */
        _onClassMouseOver(classId) {
            this.hoveredClassId = classId;
            this._refreshClasses();
        }

        /**
         * Handle class mouse out event
         * @param {number} classId
         */
        _onClassMouseOut(classId) {
            if (this.hoveredClassId === classId) {
                this.hoveredClassId = null;
            }
            this._refreshClasses();
        }

        /**
         * Handle concept click event
         * @param {number} conceptIndex
         */
        _onConceptClick(conceptIndex) {
            const wasSelected = this.selectedConceptIndex === conceptIndex;
            this.selectedConceptIndex = wasSelected ? null : conceptIndex;
            if (wasSelected) {
                this.hoveredConceptIndex = null;
            }
            this._refreshAll();
        }

        /**
         * Handle concept mouse over event
         * @param {number} conceptIndex
         */
        _onConceptMouseOver(conceptIndex) {
            this.hoveredConceptIndex = conceptIndex;
            this._refreshAll();
        }

        /**
         * Handle concept mouse out event
         * @param {number} conceptIndex
         */
        _onConceptMouseOut(conceptIndex) {
            if (this.hoveredConceptIndex === conceptIndex) {
                this.hoveredConceptIndex = null;
            }
            this._refreshAll();
        }

        /**
         * Refresh class styles
         */
        _refreshClasses() {
            for (const element of this.classElements) {
                const classId = parseInt(element.dataset.classId, 10);
                const isActive = classId === this.hoveredClassId;
                const isSelected = classId === this.selectedClassId;

                let style = StyleComputer.buildLabelStyle(null, isActive, isSelected, {
                    onClickColorMap: this.onClickColorMap,
                    enableHighlight: true,
                    showDefaultBackground: false,
                });
                if (!isActive && !isSelected) {
                    style += "outline-color: currentColor;";
                }
                element.style.cssText = style;
                element.classList.toggle("is-emphasized", isActive || isSelected);
            }
        }

        /**
         * Refresh concept list styles
         */
        _refreshConcepts() {
            if (!this.conceptElements.length) {
                return;
            }

            const activeIndex = this._getActiveConceptIndex();
            const showDefaultBackground = activeIndex === null;

            for (let i = 0; i < this.conceptElements.length; i++) {
                const element = this.conceptElements[i];
                const concept = this.topConcepts[i];
                const isActive = i === activeIndex;
                const isSelected = i === this.selectedConceptIndex;

                element.classList.remove("selected-style");

                element.style.cssText = StyleComputer.buildLabelStyle(
                    concept.color,
                    isActive,
                    isSelected,
                    {
                        onClickColorMap: this.onClickColorMap,
                        enableHighlight: true,
                        showDefaultBackground,
                    }
                );
                element.classList.toggle("is-emphasized", isActive || isSelected);

                DOMRenderer.setTooltip(
                    element,
                    StyleComputer.formatTooltip(concept.score)
                );
            }
        }

        /**
         * Refresh sample token highlights
         */
        _refreshTokens() {
            const activations = this._getActiveActivations();
            const hasTokenActivations =
                Array.isArray(activations) &&
                activations.length > 1 &&
                activations.length === this.sample.length;

            if (!hasTokenActivations || !this.topConcepts.length) {
                this._clearTokenStyles(this.sampleElements);
                return;
            }

            const activeIndex = this._getActiveConceptIndex();
            if (activeIndex === null) {
                this._updateTokensDefault(this.sampleElements, activations);
            } else {
                const concept = this.topConcepts[activeIndex];
                if (!concept) {
                    this._clearTokenStyles(this.sampleElements);
                    return;
                }
                this._updateTokensForConcept(
                    this.sampleElements,
                    activations,
                    concept
                );
            }
        }

        _updateTokensDefault(elements, activations) {
            for (let tokenIndex = 0; tokenIndex < elements.length; tokenIndex++) {
                const element = elements[tokenIndex];
                const row = activations[tokenIndex] || [];

                let bestConceptIndex = null;
                let bestValue = 0;
                let bestRawValue = 0;

                for (let i = 0; i < this.topConcepts.length; i++) {
                    const concept = this.topConcepts[i];
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

                const concept = this.topConcepts[bestConceptIndex];
                const style = StyleComputer.computeConceptStyle(
                    bestValue,
                    concept.maxAbs,
                    concept.color,
                    this.backgroundColor
                );
                element.style = style;
                DOMRenderer.setTooltip(
                    element,
                    StyleComputer.formatTooltip(bestRawValue)
                );
            }
        }

        _updateTokensForConcept(elements, activations, concept) {
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
                    this.backgroundColor
                );
                element.style = style;
                DOMRenderer.setTooltip(element, StyleComputer.formatTooltip(rawValue));
            }
        }

        _clearTokenStyles(elements) {
            for (const element of elements) {
                element.style = "";
                DOMRenderer.setTooltip(element, null);
            }
        }

        _setTopConcepts(concepts) {
            this.topConcepts = concepts;
            this.hoveredConceptIndex = null;
            this.selectedConceptIndex = null;
        }

        _computeDefaultTopConcepts() {
            if (!this.activations.length) {
                return [];
            }
            const nbConcepts = this.activations[0].length || 0;
            const totals = new Array(nbConcepts).fill(0);
            for (let tokenIndex = 0; tokenIndex < this.activations.length; tokenIndex++) {
                const row = this.activations[tokenIndex] || [];
                for (let conceptId = 0; conceptId < nbConcepts; conceptId++) {
                    const value = typeof row[conceptId] === "number" ? row[conceptId] : 0;
                    totals[conceptId] += Math.abs(value);
                }
            }
            return this._buildTopConcepts(totals, this.labels, this.activations);
        }

        _computeClassTopConcepts(classId) {
            const row = this.importances[classId] || [];
            return this._buildTopConcepts(
                row,
                this._getLabelsForClass(classId),
                this._getClassActivations(classId)
            );
        }

        _buildTopConcepts(scores, labels, activationsForMaxAbs = null) {
            const entries = [];
            for (let conceptId = 0; conceptId < scores.length; conceptId++) {
                const rawValue = typeof scores[conceptId] === "number" ? scores[conceptId] : 0;
                const rankValue = Math.abs(rawValue);
                if (rankValue === 0) {
                    continue;
                }
                entries.push({ id: conceptId, score: rawValue, rank: rankValue });
            }

            entries.sort((a, b) => b.rank - a.rank);

            const limitedEntries = this.topK > 0
                ? entries.slice(0, this.topK)
                : entries;

            return limitedEntries.map((entry) => {
                const label = Array.isArray(labels) && labels[entry.id] !== undefined
                    ? labels[entry.id]
                    : `Concept #${entry.id}`;
                const maxAbs = this._getMaxAbsForConcept(entry.id, activationsForMaxAbs);
                const color = this._getConceptColor(entry.id);
                return {
                    id: entry.id,
                    label,
                    score: entry.score,
                    maxAbs,
                    color,
                };
            });
        }

        _getMaxAbsForConcept(conceptId, activationsOverride = null) {
            let maxValue = 0;
            const activations = activationsOverride || this.activations;
            for (let tokenIndex = 0; tokenIndex < activations.length; tokenIndex++) {
                const row = activations[tokenIndex] || [];
                const rawValue = typeof row[conceptId] === "number" ? row[conceptId] : 0;
                const absValue = Math.abs(rawValue);
                if (absValue > maxValue) {
                    maxValue = absValue;
                }
            }
            return maxValue;
        }

        _getConceptColor(conceptId) {
            const mapped = StyleComputer.getColorFromMap(this.defaultColormap, conceptId);
            return mapped || this.conceptColor;
        }

        _getClassActivations(classId) {
            if (!this.activationsByClass) {
                return null;
            }
            return (
                this.activationsByClass[classId] ||
                this.activationsByClass[String(classId)] ||
                null
            );
        }

        _getActiveActivations() {
            if (this.selectedClassId !== null) {
                const classActivations = this._getClassActivations(this.selectedClassId);
                if (classActivations) {
                    return classActivations;
                }
            }
            return this.activations;
        }

        _getLabelsForClass(classId) {
            if (!this.labelsByClass) {
                return this.labels;
            }
            const key = String(classId);
            const labels = this.labelsByClass[key] || this.labelsByClass[classId];
            return Array.isArray(labels) ? labels : this.labels;
        }

        _getActiveConceptIndex() {
            if (this.hoveredConceptIndex !== null) {
                return this.hoveredConceptIndex;
            }
            if (this.selectedConceptIndex !== null) {
                return this.selectedConceptIndex;
            }
            return null;
        }
    };
})();
