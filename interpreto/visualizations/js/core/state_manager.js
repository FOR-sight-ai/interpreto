(function () {
    /**
     * StateManager - Manages visualization state and transitions
     */
    window.StateManager = class StateManager {
        constructor() {
            this.activatedClassId = null;
            this.selectedClassId = null;
            this.currentOutputId = null;
            this.selectedOutputId = null;
        }

        /**
         * Set the active class (on hover)
         * @param {number|null} classId - The class ID to activate
         * @returns {object} What changed: { classChanged: boolean }
         */
        setActiveClass(classId) {
            const changed = this.activatedClassId !== classId;
            this.activatedClassId = classId;
            return { classChanged: changed };
        }

        /**
         * Toggle the selected class (on click)
         * @param {number} classId - The class ID to toggle
         * @returns {object} What changed: { classChanged: boolean, wasDeselected: boolean }
         */
        toggleSelectedClass(classId) {
            const wasSelected = this.selectedClassId === classId;
            if (wasSelected) {
                this.selectedClassId = null;
                this.activatedClassId = null;
            } else {
                this.selectedClassId = classId;
                this.activatedClassId = classId;
            }
            return { classChanged: true, wasDeselected: wasSelected };
        }

        /**
         * Restore active class to selected class (on mouse out)
         * @returns {object} What changed: { classChanged: boolean }
         */
        restoreSelectedClass() {
            const changed = this.activatedClassId !== this.selectedClassId;
            this.activatedClassId = this.selectedClassId;
            return { classChanged: changed };
        }

        /**
         * Set the active output (on hover)
         * @param {number|null} outputId - The output ID to activate
         * @param {boolean} resetClass - Whether to reset class selection
         * @returns {object} What changed: { outputChanged: boolean, classChanged: boolean }
         */
        setActiveOutput(outputId, resetClass = false) {
            // Do not override a selected output on hover.
            if (this.selectedOutputId !== null && this.selectedOutputId !== outputId) {
                return { outputChanged: false, classChanged: false };
            }
            const outputChanged = this.currentOutputId !== outputId;
            const classChanged = resetClass && (this.activatedClassId !== null || this.selectedClassId !== null);

            this.currentOutputId = outputId;
            if (resetClass) {
                this.activatedClassId = null;
                this.selectedClassId = null;
            }

            return { outputChanged, classChanged };
        }

        /**
         * Toggle the selected output (on click)
         * @param {number} outputId - The output ID to toggle
         * @param {boolean} resetClass - Whether to reset class selection
         * @returns {object} What changed: { outputChanged: boolean, classChanged: boolean, wasDeselected: boolean }
         */
        toggleSelectedOutput(outputId, resetClass = false) {
            const wasSelected = this.selectedOutputId === outputId;
            const outputChanged = true;

            if (wasSelected) {
                this.selectedOutputId = null;
                this.currentOutputId = null;
            } else {
                this.selectedOutputId = outputId;
                this.currentOutputId = outputId;
            }

            const classChanged = resetClass;
            if (resetClass) {
                this.selectedClassId = null;
                this.activatedClassId = null;
            }

            return { outputChanged, classChanged, wasDeselected: wasSelected };
        }

        /**
         * Restore active output to selected output (on mouse out)
         * @returns {object} What changed: { outputChanged: boolean }
         */
        restoreSelectedOutput() {
            const changed = this.currentOutputId !== this.selectedOutputId;
            this.currentOutputId = this.selectedOutputId;
            return { outputChanged: changed };
        }

        /**
         * Check if an output and class are both locked (selected)
         * @returns {boolean}
         */
        isFullyLocked() {
            return this.selectedOutputId !== null && this.selectedClassId !== null;
        }

        /**
         * Check if a class is selected
         * @returns {boolean}
         */
        hasSelectedClass() {
            return this.selectedClassId !== null;
        }

        /**
         * Check if an output is selected
         * @returns {boolean}
         */
        hasSelectedOutput() {
            return this.selectedOutputId !== null;
        }

        /**
         * Get current state snapshot
         * @returns {object}
         */
        getState() {
            return {
                activatedClassId: this.activatedClassId,
                selectedClassId: this.selectedClassId,
                currentOutputId: this.currentOutputId,
                selectedOutputId: this.selectedOutputId,
            };
        }

        /**
         * Log current state for debugging
         * @param {string} prefix - Log prefix
         */
        trace(prefix = "") {
            console.log(
                `\t[${prefix}]` +
                `\tclass: selected=${this.selectedClassId}/activated=${this.activatedClassId}` +
                `\toutput: selected=${this.selectedOutputId}/current=${this.currentOutputId}`
            );
        }
    };
})();
