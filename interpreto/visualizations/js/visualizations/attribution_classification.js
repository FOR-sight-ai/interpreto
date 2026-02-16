(function () {
    /**
     * ClassificationVisualization - Visualization for classification tasks
     * Handles both single-class and multi-class attribution display
     */
    window.ClassificationVisualization = class ClassificationVisualization {
        /**
         * @param {string} uniqueIdClasses - The unique id of the div containing the classes
         * @param {string} uniqueIdInputs - The unique id of the div containing the inputs
         * @param {boolean} highlightBorder - Whether to highlight the border of the words
         * @param {string} jsonData - The JSON data containing classes and inputs
         */
        constructor(uniqueIdClasses, uniqueIdInputs, highlightBorder, jsonData) {
            console.log("Creating ClassificationVisualization");

            this.uniqueIdClasses = uniqueIdClasses;
            this.uniqueIdInputs = uniqueIdInputs;
            this.highlightBorder = highlightBorder === "True";
            this.data = JSON.parse(jsonData);
            this.onClickColorMap =
                Array.isArray(this.data.onclick_colormap) &&
                this.data.onclick_colormap.length >= 2
                    ? this.data.onclick_colormap
                    : null;

            // State management
            this.state = new StateManager();

            // Determine if single or multi-class
            this.isMultiClass = this.data.classes.length > 1;

            // DOM element references
            this.classElements = [];
            this.inputWordElements = [];

            // Initialize state
            this.state.currentOutputId = 0; // Always 0 for classification
            if (!this.isMultiClass) {
                this.state.activatedClassId = 0;
                this.state.selectedClassId = 0;
            }
            // For multi-class: activatedClassId and selectedClassId remain null initially
            // This triggers the default multi-class view

            // Create DOM
            this._createClasses();
            this._createInputs();

            // Initial render
            this._refreshAll();
        }

        /**
         * Create class elements in the DOM
         */
        _createClasses() {
            const container = document.getElementById(this.uniqueIdClasses);
            if (!container) return;

            if (this.isMultiClass) {
                this.classElements = DOMRenderer.renderClassButtonsWithColors(
                    container,
                    this.data.classes,
                    {
                        onClick: (id) => this._onClassClick(id),
                        onMouseOver: (id) => this._onClassMouseOver(id),
                        onMouseOut: (id) => this._onClassMouseOut(id),
                    }
                );
            } else {
                const element = DOMRenderer.renderClassLabel(container, this.data.classes[0]);
                this.classElements = [element];
            }
        }

        /**
         * Create input elements in the DOM
         */
        _createInputs() {
            const container = document.getElementById(this.uniqueIdInputs);
            if (!container) return;

            const { wordElements } = DOMRenderer.renderInputs(container, this.data.inputs.words);
            this.inputWordElements = wordElements;
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

            // Only activate if no class is selected (allow hover preview)
            // Or if a class is already selected (allow switching preview)
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
            this._refreshInputs();
        }

        /**
         * Refresh class elements styles
         */
        _refreshClasses() {
            if (this.isMultiClass) {
                ViewUpdater.updateClasses(this.classElements, this.state.getState(), {
                    highlightActiveText: true,
                    useSelectedStyle: false,
                    onClickColorMap: this.onClickColorMap,
                });
            }
        }

        /**
         * Refresh input elements styles
         */
        _refreshInputs() {
            if (this.isMultiClass && this.state.activatedClassId === null) {
                // Default multi-class view: show dominant class colors
                ViewUpdater.updateInputsDefaultMultiClass(
                    this.inputWordElements,
                    this.state.getState(),
                    this.data,
                    { highlightBorder: this.highlightBorder }
                );
            } else {
                // Single class selected/hovered: show that class's attributions
                ViewUpdater.updateInputs(
                    this.inputWordElements,
                    this.state.getState(),
                    this.data,
                    { highlightBorder: this.highlightBorder }
                );
            }
        }
    };
})();
