(function () {
    /**
     * GenerationLocalConceptsVisualization - Visualization for local concept activations
     */
    window.GenerationLocalConceptsVisualization = class GenerationLocalConceptsVisualization {
        /**
         * @param {string} uniqueIdSample - The unique id of the div containing the sample
         * @param {string} uniqueIdConcepts - The unique id of the div containing the concepts
         * @param {string} uniqueIdConceptsWrapper - The unique id of the concepts wrapper div
         * @param {string} jsonData - The JSON data containing the sample and activations
         */
        constructor(
            uniqueIdSample,
            uniqueIdConcepts,
            uniqueIdConceptsWrapper,
            jsonData
        ) {
            console.log("Creating GenerationLocalConceptsVisualization");

            this.uniqueIdSample = uniqueIdSample;
            this.uniqueIdConcepts = uniqueIdConcepts;
            this.uniqueIdConceptsWrapper = uniqueIdConceptsWrapper;
            this.data = JSON.parse(jsonData);

            this.sample = Array.isArray(this.data.sample) ? this.data.sample : [];
            this.activations = Array.isArray(this.data.activations)
                ? this.data.activations
                : [];
            this.importances = Array.isArray(this.data.importances)
                ? this.data.importances
                : [];
            this.labels = Array.isArray(this.data.labels) ? this.data.labels : [];
            this.outputStartIndex = Math.max(
                0,
                this.sample.length - this.importances.length
            );

            this.sampleElements = [];
            this.conceptElements = [];
            this.topConcepts = [];

            this.topK = Math.max(0, parseInt(this.data.top_k || 0, 10));
            this.conceptColor = this.data.concept_color || "#f39c12";
            this.backgroundColor = StyleComputer.getBackgroundRgb();
            this.defaultColormap = this.data.default_colormap || {};
            this.onClickColorMap =
                Array.isArray(this.data.onclick_colormap) &&
                this.data.onclick_colormap.length >= 2
                    ? this.data.onclick_colormap
                    : null;

            this.state = new StateManager();

            // Create DOM
            this._createSample();
            this._setTopConcepts(this._computeDefaultTopConcepts());
            this._renderConcepts();

            // Initial render
            this._refreshAll();
        }

        /**
         * Create sample elements in the DOM
         */
        _createSample() {
            const container = document.getElementById(this.uniqueIdSample);
            if (!container) return;

            const { wordElements } = DOMRenderer.renderInputs(container, this.sample);
            this.sampleElements = wordElements;

            const outputCount = this.importances.length;
            for (let i = this.outputStartIndex; i < this.sampleElements.length; i++) {
                const outputIndex = i - this.outputStartIndex;
                if (outputIndex < 0 || outputIndex >= outputCount) {
                    continue;
                }
                const element = this.sampleElements[i];
                element.classList.add("reactive-word-style");
                element.addEventListener("click", (e) => {
                    e.preventDefault();
                    this._onOutputClick(outputIndex);
                });
            }
        }

        /**
         * Render concept elements in the DOM
         */
        _renderConcepts() {
            const wrapper = document.getElementById(this.uniqueIdConceptsWrapper);
            const container = document.getElementById(this.uniqueIdConcepts);
            if (!wrapper || !container) {
                this.conceptElements = [];
                return;
            }

            this.conceptElements = ConceptsLocalUtils.renderConceptList(
                wrapper,
                container,
                this.topConcepts,
                {
                    onClick: (index) => this._onConceptClick(index),
                    onMouseOver: (index) => this._onConceptMouseOver(index),
                    onMouseOut: (index) => this._onConceptMouseOut(index),
                }
            );
        }

        /**
         * Refresh all view components
         */
        _refreshAll() {
            this._refreshSampleSelections();
            this._refreshConcepts();
            this._refreshTokens();
        }

        /**
         * Handle output click event
         * @param {number} outputIndex
         */
        _onOutputClick(outputIndex) {
            const wasSelected = this.state.selectedOutputId === outputIndex;
            this.state.toggleSelectedOutput(outputIndex, true);

            if (wasSelected) {
                this._setTopConcepts(this._computeDefaultTopConcepts());
            } else {
                this._setTopConcepts(this._computeOutputTopConcepts(outputIndex));
            }

            this._renderConcepts();
            this._refreshAll();
        }

        /**
         * Handle concept click event
         * @param {number} conceptIndex
         */
        _onConceptClick(conceptIndex) {
            this.state.toggleSelectedClass(conceptIndex);
            this._refreshAll();
        }

        /**
         * Handle concept mouse over event
         * @param {number} conceptIndex
         */
        _onConceptMouseOver(conceptIndex) {
            this.state.setActiveClass(conceptIndex);
            this._refreshAll();
        }

        /**
         * Handle concept mouse out event
         * @param {number} conceptIndex
         */
        _onConceptMouseOut(conceptIndex) {
            if (this.state.activatedClassId === conceptIndex) {
                this.state.restoreSelectedClass();
                this._refreshAll();
            }
        }

        /**
         * Refresh sample styles for output selection
         */
        _refreshSampleSelections() {
            const outputCount = this.importances.length;
            for (let i = 0; i < this.sampleElements.length; i++) {
                const element = this.sampleElements[i];
                const outputIndex = i - this.outputStartIndex;
                const isOutput = outputIndex >= 0 && outputIndex < outputCount;
                if (!isOutput) {
                    element.classList.remove("selected-style");
                    continue;
                }
                const isSelected = outputIndex === this.state.selectedOutputId;
                element.classList.toggle("selected-style", isSelected);
            }
        }

        /**
         * Refresh concept list styles
         */
        _refreshConcepts() {
            if (!this.conceptElements.length) {
                return;
            }

            const showDefaultBackground = this.state.activatedClassId === null;

            for (let i = 0; i < this.conceptElements.length; i++) {
                const element = this.conceptElements[i];
                const concept = this.topConcepts[i];
                const isActive = i === this.state.activatedClassId;
                const isSelected = i === this.state.selectedClassId;

                element.classList.remove("selected-style");

                element.style.cssText = StyleComputer.buildLabelStyle(
                    concept.color,
                    isActive,
                    isSelected,
                    {
                        onClickColorMap: this.onClickColorMap,
                        enableHighlight: true,
                        showDefaultBackground,
                        showBackgroundForSelected: true,
                        useOnClickColorMap: false,
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
            if (!this.topConcepts.length) {
                ConceptsLocalUtils.clearTokenStyles(this.sampleElements);
                return;
            }

            if (this.state.activatedClassId === null) {
                ConceptsLocalUtils.updateTokensDefault(
                    this.sampleElements,
                    this.activations,
                    this.topConcepts,
                    this.backgroundColor
                );
            } else {
                const concept = this.topConcepts[this.state.activatedClassId];
                if (!concept) {
                    ConceptsLocalUtils.clearTokenStyles(this.sampleElements);
                    return;
                }
                ConceptsLocalUtils.updateTokensForConcept(
                    this.sampleElements,
                    this.activations,
                    concept,
                    this.backgroundColor
                );
            }
        }

        _setTopConcepts(concepts) {
            this.topConcepts = concepts;
            this.state.activatedClassId = null;
            this.state.selectedClassId = null;
        }

        _computeDefaultTopConcepts() {
            if (!this.importances.length || this.topK <= 0) {
                return [];
            }
            const aggregated = this._aggregateImportances();
            return this._buildTopConcepts(aggregated);
        }

        _computeOutputTopConcepts(outputIndex) {
            if (!this.importances.length || this.topK <= 0) {
                return [];
            }
            const row = this.importances[outputIndex] || [];
            return this._buildTopConcepts(row);
        }

        _aggregateImportances() {
            if (!this.importances.length) {
                return [];
            }
            const nbConcepts = this.importances[0].length || 0;
            const totals = new Array(nbConcepts).fill(0);
            for (let outputIndex = 0; outputIndex < this.importances.length; outputIndex++) {
                const row = this.importances[outputIndex] || [];
                for (let conceptId = 0; conceptId < nbConcepts; conceptId++) {
                    const value = typeof row[conceptId] === "number" ? row[conceptId] : 0;
                    totals[conceptId] += Math.abs(value);
                }
            }
            return totals;
        }

        _buildTopConcepts(scores) {
            const entries = ConceptsLocalUtils.buildRankedEntries(scores);
            const limit = Math.min(this.topK, entries.length);
            const selected = entries.slice(0, limit);

            return selected.map((entry) => {
                const label = this.labels[entry.id] || `Concept #${entry.id}`;
                const maxAbs = ConceptsLocalUtils.getMaxAbsForConcept(entry.id, this.activations);
                const color = ConceptsLocalUtils.getConceptColor(
                    entry.id,
                    this.defaultColormap,
                    this.conceptColor
                );
                return {
                    id: entry.id,
                    label,
                    score: entry.absValue,
                    maxAbs,
                    color,
                };
            });
        }

    };
})();
