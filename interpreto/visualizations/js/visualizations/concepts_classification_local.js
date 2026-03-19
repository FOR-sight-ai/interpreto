(function () {
    /**
     * ClassificationLocalConceptsVisualization - Static visualization for local concepts in
     * classification tasks.
     */
    window.ClassificationLocalConceptsVisualization = class ClassificationLocalConceptsVisualization {
        /**
         * @param {string} uniqueIdRoot - The unique id of the root div containing the visualization
         * @param {string} jsonData - The JSON data containing classes, concepts, and sample
         */
        constructor(uniqueIdRoot, jsonData) {
            console.log("Creating ClassificationLocalConceptsVisualization");

            this.uniqueIdRoot = uniqueIdRoot;
            this.data = JSON.parse(jsonData);

            this.classes = Array.isArray(this.data.classes) ? this.data.classes : [];
            this.sample = Array.isArray(this.data.sample) ? this.data.sample : [];
            this.importances = Array.isArray(this.data.importances)
                ? this.data.importances
                : [];
            this.labels = Array.isArray(this.data.labels) ? this.data.labels : [];
            this.labelsByClass = this.data.labels_by_class || null;
            this.topK = Math.max(0, parseInt(this.data.top_k || 0, 10));

            this.positiveColor = getComputedStyle(document.documentElement)
                .getPropertyValue("--positive-color")
                .trim() || "#ff0000";
            this.negativeColor = getComputedStyle(document.documentElement)
                .getPropertyValue("--negative-color")
                .trim() || "#0000ff";

            this.maxAbsImportance = this._computeMaxAbsImportance();
            this.unionConceptIds = this._computeUnionConceptIds();

            this._render();
        }

        _render() {
            const root = document.getElementById(this.uniqueIdRoot);
            if (!root) {
                return;
            }

            root.innerHTML = "";

            const plot = document.createElement("div");
            plot.classList.add("local-classification-plot");

            plot.appendChild(this._createDivider());
            plot.appendChild(this._createSampleRow());
            plot.appendChild(this._createDivider());
            plot.appendChild(this._createHeaderRow());

            for (let classId = 0; classId < this.classes.length; classId++) {
                plot.appendChild(this._createClassRow(classId));
            }

            plot.appendChild(this._createDivider());
            root.appendChild(plot);
        }

        _createDivider() {
            const divider = document.createElement("div");
            divider.classList.add("local-classification-divider");
            return divider;
        }

        _createSampleRow() {
            const row = document.createElement("div");
            row.classList.add("local-classification-grid", "local-classification-sample-row");

            const label = document.createElement("div");
            label.classList.add("local-classification-side-label");
            label.textContent = "Sample";

            const content = document.createElement("div");
            content.classList.add("local-classification-main-cell");

            const line = document.createElement("div");
            line.classList.add("local-classification-token-line");

            for (let tokenIndex = 0; tokenIndex < this.sample.length; tokenIndex++) {
                const tokenElement = document.createElement("div");
                tokenElement.classList.add("common-word-style", "local-classification-token");
                tokenElement.dataset.wordIndex = tokenIndex.toString();
                tokenElement.textContent = DOMRenderer.normalizeSpecialChars(this.sample[tokenIndex]);
                line.appendChild(tokenElement);
            }

            content.appendChild(line);
            row.appendChild(label);
            row.appendChild(content);
            return row;
        }

        _createHeaderRow() {
            const row = document.createElement("div");
            row.classList.add("local-classification-grid", "local-classification-header-row");

            const classesHeader = document.createElement("div");
            classesHeader.classList.add("local-classification-column-heading");
            classesHeader.textContent = "Classes";

            const conceptsHeader = document.createElement("div");
            conceptsHeader.classList.add("local-classification-column-heading");
            conceptsHeader.textContent = "Concepts";

            row.appendChild(classesHeader);
            row.appendChild(conceptsHeader);
            return row;
        }

        _createClassRow(classId) {
            const row = document.createElement("div");
            row.classList.add("local-classification-grid", "local-classification-class-row");

            const classCell = document.createElement("div");
            classCell.classList.add(
                "common-word-style",
                "local-classification-class-name"
            );
            classCell.dataset.classId = classId.toString();
            classCell.textContent = this._getClassName(classId);

            const conceptsCell = document.createElement("div");
            conceptsCell.classList.add("local-classification-main-cell");

            const conceptsLine = document.createElement("div");
            conceptsLine.classList.add("local-classification-concepts-line");

            const rowConcepts = this._buildRowConcepts(classId);
            for (let conceptIndex = 0; conceptIndex < rowConcepts.length; conceptIndex++) {
                conceptsLine.appendChild(this._createConceptChip(rowConcepts[conceptIndex]));
            }

            conceptsCell.appendChild(conceptsLine);
            row.appendChild(classCell);
            row.appendChild(conceptsCell);
            return row;
        }

        _createConceptChip(concept) {
            const chip = document.createElement("div");
            chip.classList.add(
                "common-word-style",
                "highlighted-word-style",
                "concept-style",
                "local-classification-chip"
            );
            chip.dataset.conceptId = concept.id.toString();
            chip.textContent = this._formatLabel(concept.label);
            chip.style.cssText = this._buildConceptChipStyle(concept.score);
            DOMRenderer.setTooltip(chip, StyleComputer.formatTooltip(concept.score));
            return chip;
        }

        _computeMaxAbsImportance() {
            let maxValue = 0;
            for (let classId = 0; classId < this.importances.length; classId++) {
                const row = Array.isArray(this.importances[classId]) ? this.importances[classId] : [];
                for (let conceptId = 0; conceptId < row.length; conceptId++) {
                    const rawValue = typeof row[conceptId] === "number" ? row[conceptId] : 0;
                    const absValue = Math.abs(rawValue);
                    if (absValue > maxValue) {
                        maxValue = absValue;
                    }
                }
            }
            return maxValue;
        }

        _computeUnionConceptIds() {
            const conceptIds = [];
            const seen = new Set();

            for (let classId = 0; classId < this.importances.length; classId++) {
                const rankedEntries = this._buildAbsoluteRankedEntries(this.importances[classId] || []);
                const selectedEntries = rankedEntries.slice(0, this.topK);
                for (let entryIndex = 0; entryIndex < selectedEntries.length; entryIndex++) {
                    const conceptId = selectedEntries[entryIndex].id;
                    if (seen.has(conceptId)) {
                        continue;
                    }
                    seen.add(conceptId);
                    conceptIds.push(conceptId);
                }
            }

            return conceptIds;
        }

        _buildAbsoluteRankedEntries(scores) {
            const entries = [];
            for (let conceptId = 0; conceptId < scores.length; conceptId++) {
                const rawValue = typeof scores[conceptId] === "number" ? scores[conceptId] : 0;
                entries.push({
                    id: conceptId,
                    rawValue,
                    absValue: Math.abs(rawValue),
                });
            }

            entries.sort((left, right) => {
                if (right.absValue !== left.absValue) {
                    return right.absValue - left.absValue;
                }
                if (right.rawValue !== left.rawValue) {
                    return right.rawValue - left.rawValue;
                }
                return left.id - right.id;
            });

            return entries;
        }

        _getRowThreshold(classId) {
            const scores = Array.isArray(this.importances[classId]) ? this.importances[classId] : [];
            if (!scores.length || this.topK <= 0) {
                return Infinity;
            }

            const thresholdIndex = Math.min(this.topK, scores.length) - 1;
            const absoluteScores = scores
                .map((value) => Math.abs(typeof value === "number" ? value : 0))
                .sort((left, right) => right - left);
            return absoluteScores[thresholdIndex];
        }

        _buildRowConcepts(classId) {
            const threshold = this._getRowThreshold(classId);
            const scores = Array.isArray(this.importances[classId]) ? this.importances[classId] : [];
            const labels = this._getLabelsForClass(classId);
            const concepts = [];

            for (let conceptIndex = 0; conceptIndex < this.unionConceptIds.length; conceptIndex++) {
                const conceptId = this.unionConceptIds[conceptIndex];
                const rawValue = typeof scores[conceptId] === "number" ? scores[conceptId] : 0;
                if (Math.abs(rawValue) < threshold) {
                    continue;
                }
                concepts.push({
                    id: conceptId,
                    label: labels[conceptId] !== undefined ? labels[conceptId] : `Concept #${conceptId}`,
                    score: rawValue,
                });
            }

            concepts.sort((left, right) => {
                if (right.score !== left.score) {
                    return right.score - left.score;
                }
                return left.id - right.id;
            });

            return concepts;
        }

        _getClassName(classId) {
            const classData = this.classes[classId] || {};
            return classData.name || `Class #${classId}`;
        }

        _getLabelsForClass(classId) {
            if (!this.labelsByClass) {
                return this.labels;
            }
            const key = String(classId);
            const labels = this.labelsByClass[key] || this.labelsByClass[classId];
            return Array.isArray(labels) ? labels : this.labels;
        }

        _buildConceptChipStyle(score) {
            const absValue = Math.abs(score);
            if (absValue === 0 || this.maxAbsImportance <= 0) {
                return "background-color: transparent; outline-color: transparent;";
            }

            const colorHex = score >= 0 ? this.positiveColor : this.negativeColor;
            const color = StyleComputer.hexToRgb(colorHex);
            const normalizedAlpha = Math.max(
                0,
                Math.min(absValue / this.maxAbsImportance, 1)
            );
            const brightness = StyleComputer.getBrightness(color);

            let style =
                `background-color: rgba(${color[0]},${color[1]},${color[2]},${normalizedAlpha});` +
                `outline-color: rgba(${color[0]},${color[1]},${color[2]},${normalizedAlpha});`;

            if (normalizedAlpha >= 0.35 && brightness < 150) {
                style += "color: white;";
            }

            return style;
        }

        // This mirrors DOMRenderer.renderConcepts label formatting and could be abstracted later.
        _formatLabel(label) {
            if (Array.isArray(label)) {
                return label.join("\n");
            }
            return String(label);
        }
    };
})();
