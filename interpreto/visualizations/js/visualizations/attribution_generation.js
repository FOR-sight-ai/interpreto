(function () {
    /**
     * GenerationVisualization - Visualization for generation tasks
     * Handles attribution display for input and output tokens in generative models
     */
    window.GenerationVisualization = class GenerationVisualization {
        /**
         * @param {string} uniqueIdInputs - The unique id of the div containing the inputs
         * @param {string} uniqueIdOutputs - The unique id of the div containing the outputs
         * @param {boolean} highlightBorder - Whether to highlight the border of the words
         * @param {string} jsonData - The JSON data containing classes, inputs and outputs
         */
        constructor(uniqueIdInputs, uniqueIdOutputs, highlightBorder, jsonData) {
            console.log("Creating GenerationVisualization");

            this.uniqueIdInputs = uniqueIdInputs;
            this.uniqueIdOutputs = uniqueIdOutputs;
            this.highlightBorder = highlightBorder === "True";
            this.data = JSON.parse(jsonData);

            // State management
            this.state = new StateManager();

            // DOM element references
            this.inputWordElements = [];
            this.outputElements = [];

            // Initialize state - generation has a single implicit class (index 0)
            this.state.activatedClassId = 0;
            this.state.selectedClassId = 0;
            this.state.currentOutputId = null;

            // Create DOM
            this._createInputs();
            this._createOutputs();

            // Initial render
            this._refreshAll();
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
         * Create output elements in the DOM
         */
        _createOutputs() {
            const container = document.getElementById(this.uniqueIdOutputs);
            if (!container) return;

            this.outputElements = DOMRenderer.renderOutputs(
                container,
                this.data.outputs.words,
                {
                    onClick: (id) => this._onOutputClick(id),
                    onMouseOver: (id) => this._onOutputMouseOver(id),
                    onMouseOut: (id) => this._onOutputMouseOut(id),
                }
            );
        }

        /**
         * Handle output click event
         * @param {number} outputId
         */
        _onOutputClick(outputId) {
            console.log("Output clicked:", outputId);
            this.state.trace("before click");

            this.state.toggleSelectedOutput(outputId, false);

            this.state.trace("after click");
            this._refreshAll();
        }

        /**
         * Handle output mouse over event
         * @param {number} outputId
         */
        _onOutputMouseOver(outputId) {
            console.log("Output hover:", outputId);

            this.state.setActiveOutput(outputId, false);
            this._refreshAll();
        }

        /**
         * Handle output mouse out event
         * @param {number} outputId
         */
        _onOutputMouseOut(outputId) {
            console.log("Output mouse out:", outputId);

            this.state.restoreSelectedOutput();
            this._refreshAll();
        }

        /**
         * Refresh all visual elements based on current state
         */
        _refreshAll() {
            this._refreshInputs();
            this._refreshOutputs();
        }

        /**
         * Refresh input word highlighting based on current state
         */
        _refreshInputs() {
            ViewUpdater.updateInputs(
                this.inputWordElements,
                this.state.getState(),
                this.data,
                { highlightBorder: this.highlightBorder }
            );
        }

        /**
         * Refresh output element states based on current state
         */
        _refreshOutputs() {
            ViewUpdater.updateOutputs(
                this.outputElements,
                this.state.getState(),
                this.data,
                { highlightBorder: this.highlightBorder }
            );
        }
    };
})();
