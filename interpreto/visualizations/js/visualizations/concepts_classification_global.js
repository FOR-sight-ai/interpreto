(function () {
    /**
     * ClassificationConceptsVisualization - Visualization for global concept importances
     */
    window.ClassificationConceptsVisualization = class ClassificationConceptsVisualization {
        /**
         * @param {string} uniqueIdClasses - The unique id of the div containing the classes
         * @param {string} uniqueIdConcepts - The unique id of the div containing the concepts
         * @param {string} uniqueIdConceptsWrapper - The unique id of the concepts wrapper div
         * @param {string} jsonData - The JSON data containing classes and concepts
         */
        constructor(uniqueIdClasses, uniqueIdConcepts, uniqueIdConceptsWrapper, jsonData) {
            console.log("Creating ClassificationConceptsVisualization");

            this.uniqueIdClasses = uniqueIdClasses;
            this.uniqueIdConcepts = uniqueIdConcepts;
            this.uniqueIdConceptsWrapper = uniqueIdConceptsWrapper;
            this.data = JSON.parse(jsonData);

            // State management
            this.state = new StateManager();

            // DOM element references
            this.classElements = [];
            this.conceptElements = [];
            this.currentConceptClassId = null;

            this.conceptColor = this.data.concept_color || "#f39c12";
            this.onClickColorMap =
                Array.isArray(this.data.onclick_colormap) &&
                this.data.onclick_colormap.length >= 2
                    ? this.data.onclick_colormap
                    : null;

            // Create DOM
            this._createClasses();

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
                this.data.classes,
                {
                    onClick: (id) => this._onClassClick(id),
                    onMouseOver: (id) => this._onClassMouseOver(id),
                    onMouseOut: (id) => this._onClassMouseOut(id),
                },
                { useColors: true }
            );
        }

        /**
         * Handle class click event
         * @param {number} classId
         */
        _onClassClick(classId) {
            console.log("Class clicked:", classId);
            this.state.trace("before click");

            this.state.toggleSelectedClass(classId);
            this._refreshAll();
        }

        /**
         * Handle class mouse over event
         * @param {number} classId
         */
        _onClassMouseOver(classId) {
            console.log("Class hover:", classId);

            this.state.setActiveClass(classId);
            this._refreshAll();
        }

        /**
         * Handle class mouse out event
         * @param {number} classId
         */
        _onClassMouseOut(classId) {
            console.log("Class mouse out:", classId);

            this.state.restoreSelectedClass();
            this._refreshAll();
        }

        /**
         * Refresh all view components
         */
        _refreshAll() {
            this._refreshClasses();
            this._refreshConcepts();
        }

        /**
         * Refresh class elements styles
         */
        _refreshClasses() {
            ViewUpdater.updateClasses(this.classElements, this.state.getState(), {
                highlightActiveText: true,
                useSelectedStyle: false,
                onClickColorMap: this.onClickColorMap,
            });
        }

        /**
         * Refresh concept elements and styles
         */
        _refreshConcepts() {
            const wrapper = document.getElementById(this.uniqueIdConceptsWrapper);
            const container = document.getElementById(this.uniqueIdConcepts);
            if (!wrapper || !container) return;

            if (this.state.activatedClassId === null) {
                wrapper.classList.add("is-hidden");
                container.innerHTML = "";
                this.conceptElements = [];
                this.currentConceptClassId = null;
                return;
            }

            wrapper.classList.remove("is-hidden");

            if (this.currentConceptClassId !== this.state.activatedClassId) {
                container.innerHTML = "";
                const concepts = this.data.concepts[this.state.activatedClassId] || [];
                const { conceptElements } = DOMRenderer.renderConcepts(container, concepts);
                this.conceptElements = conceptElements;
                this.currentConceptClassId = this.state.activatedClassId;
            }

            this._updateConceptStyles();
        }

        /**
         * Update concept styles based on importance
         */
        _updateConceptStyles() {
            if (this.currentConceptClassId === null) return;
            const concepts = this.data.concepts[this.currentConceptClassId] || [];
            const classMeta = this.data.classes[this.currentConceptClassId] || {};
            const classColor = classMeta.color || this.conceptColor;

            let maxImportance = 0;
            for (const concept of concepts) {
                const rawValue = typeof concept.importance === "number"
                    ? concept.importance
                    : 0;
                const value = Math.abs(rawValue);
                if (value > maxImportance) {
                    maxImportance = value;
                }
            }

            for (let i = 0; i < this.conceptElements.length; i++) {
                const element = this.conceptElements[i];
                const concept = concepts[i];
                const rawValue = concept && typeof concept.importance === "number"
                    ? concept.importance
                    : 0;
                const value = Math.abs(rawValue);
                element.style.cssText = StyleComputer.computeDefaultClassStyle(
                    value,
                    classColor,
                    maxImportance
                );
                DOMRenderer.setTooltip(
                    element,
                    StyleComputer.formatTooltip(rawValue)
                );
            }
        }

    };
})();
