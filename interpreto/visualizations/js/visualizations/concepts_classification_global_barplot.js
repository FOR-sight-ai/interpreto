(function () {
    /**
     * ClassificationConceptsBarPlotVisualization - Horizontal bar plot for global concept importances
     */
    window.ClassificationConceptsBarPlotVisualization = class ClassificationConceptsBarPlotVisualization {
        /**
         * @param {string} uniqueIdClasses - The unique id of the div containing the classes
         * @param {string} uniqueIdConcepts - The unique id of the div containing the concepts
         * @param {string} uniqueIdConceptsWrapper - The unique id of the concepts wrapper div
         * @param {string} uniqueIdConceptsScale - The unique id of the concepts scale div
         * @param {string} jsonData - The JSON data containing classes and concepts
         */
        constructor(
            uniqueIdClasses,
            uniqueIdConcepts,
            uniqueIdConceptsWrapper,
            uniqueIdConceptsScale,
            jsonData
        ) {
            console.log("Creating ClassificationConceptsBarPlotVisualization");

            this.uniqueIdClasses = uniqueIdClasses;
            this.uniqueIdConcepts = uniqueIdConcepts;
            this.uniqueIdConceptsWrapper = uniqueIdConceptsWrapper;
            this.uniqueIdConceptsScale = uniqueIdConceptsScale;
            this.data = JSON.parse(jsonData);

            // State management
            this.selectedClassIds = new Set();
            this.hoveredClassId = null;
            this.currentDisplayKey = null;

            this.classElements = [];

            this.conceptColor = this.data.concept_color || "#f39c12";
            this.onClickColorMap =
                Array.isArray(this.data.onclick_colormap) &&
                this.data.onclick_colormap.length >= 2
                    ? this.data.onclick_colormap
                    : null;
            this.conceptsAreClasswise = !!this.data.concepts_are_classwise;

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
                this.data.classes || [],
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
            if (this.selectedClassIds.has(classId)) {
                this.selectedClassIds.delete(classId);
            } else {
                this.selectedClassIds.add(classId);
            }
            this._refreshAll();
        }

        /**
         * Handle class mouse over event
         * @param {number} classId
         */
        _onClassMouseOver(classId) {
            this.hoveredClassId = classId;
            this._refreshAll();
        }

        /**
         * Handle class mouse out event
         * @param {number} classId
         */
        _onClassMouseOut(classId) {
            if (this.hoveredClassId === classId) {
                this.hoveredClassId = null;
            }
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
            const backgroundRgb = StyleComputer.getBackgroundRgb();

            for (const element of this.classElements) {
                const classId = parseInt(element.dataset.classId, 10);
                const classColor = element.dataset.classColor || null;
                const isSelected = this.selectedClassIds.has(classId);
                const isActive = classId === this.hoveredClassId;

                const showBackground = isSelected && classColor;
                const textColor = showBackground && classColor
                    ? StyleComputer.getReadableTextColor(
                        StyleComputer.hexToRgb(classColor)
                    )
                    : StyleComputer.getReadableTextColor(backgroundRgb);

                let outlineColor = "transparent";
                if (isActive && !isSelected) {
                    outlineColor = classColor || "currentColor";
                }

                element.style.cssText =
                    `background-color: ${showBackground ? classColor : "transparent"};` +
                    "text-shadow: none;" +
                    `color: ${textColor};` +
                    `outline-color: ${outlineColor};`;

                element.classList.toggle("is-emphasized", isSelected || isActive);
            }
        }

        /**
         * Refresh concept bars
         */
        _refreshConcepts() {
            const wrapper = document.getElementById(this.uniqueIdConceptsWrapper);
            const container = document.getElementById(this.uniqueIdConcepts);
            if (!wrapper || !container) return;

            const displayedClassIds = this._getDisplayedClassIds();
            if (!displayedClassIds.length) {
                wrapper.classList.add("is-hidden");
                container.innerHTML = "";
                this.currentDisplayKey = null;
                this._renderScale(0);
                return;
            }

            const key = displayedClassIds.join(",");
            if (key === this.currentDisplayKey) {
                wrapper.classList.remove("is-hidden");
                return;
            }

            this.currentDisplayKey = key;
            wrapper.classList.remove("is-hidden");
            container.innerHTML = "";

            const groups = this._buildConceptGroups(displayedClassIds);

            const plot = document.createElement("div");
            plot.classList.add("concept-barplot");
            container.appendChild(plot);

            let globalMax = 0;
            for (const group of groups) {
                for (const value of group.values) {
                    if (value.absValue > globalMax) {
                        globalMax = value.absValue;
                    }
                }
            }

            this._renderScale(globalMax);

            for (const group of groups) {
                const groupElement = document.createElement("div");
                groupElement.classList.add("concept-barplot-group");
                groupElement.dataset.conceptId = String(group.id);

                const label = document.createElement("div");
                label.classList.add("concept-barplot-label");
                label.textContent = this._formatLabel(group.label, group.id);
                groupElement.appendChild(label);

                const bars = document.createElement("div");
                bars.classList.add("concept-barplot-bars");

                for (const value of group.values) {
                    const classMeta = this.data.classes[value.classId] || {};
                    const classColor = classMeta.color || this.conceptColor;
                    const ratio = globalMax > 0 ? value.absValue / globalMax : 0;
                    const widthPercent = Math.max(0, Math.min(ratio, 1)) * 50;

                    const track = document.createElement("div");
                    track.classList.add("concept-barplot-bar-track", "highlighted-word-style");

                    const fill = document.createElement("div");
                    fill.classList.add("concept-barplot-bar-fill");
                    fill.style.width = `${widthPercent}%`;
                    if (value.rawValue < 0) {
                        fill.classList.add("is-negative");
                        fill.style.right = "50%";
                        fill.style.left = "auto";
                    } else {
                        fill.classList.add("is-positive");
                        fill.style.left = "50%";
                        fill.style.right = "auto";
                    }
                    fill.style.backgroundColor = classColor;
                    if (ratio > 0) {
                        fill.style.minWidth = "2px";
                    }

                    track.appendChild(fill);
                    bars.appendChild(track);

                    const className = classMeta.name || `Class ${value.classId}`;
                    DOMRenderer.setTooltip(
                        track,
                        `${className}: ${StyleComputer.formatTooltip(value.rawValue)}`
                    );
                }

                groupElement.appendChild(bars);
                plot.appendChild(groupElement);
            }
        }

        _getDisplayedClassIds() {
            if (this.selectedClassIds.size) {
                const ordered = [];
                const classCount = Array.isArray(this.data.classes)
                    ? this.data.classes.length
                    : 0;
                for (let i = 0; i < classCount; i++) {
                    if (this.selectedClassIds.has(i)) {
                        ordered.push(i);
                    }
                }
                return ordered;
            }
            if (this.hoveredClassId !== null) {
                return [this.hoveredClassId];
            }
            return [];
        }

        _buildConceptGroups(classIds) {
            const map = new Map();

            for (const classId of classIds) {
                const concepts = Array.isArray(this.data.concepts)
                    ? this.data.concepts[classId] || []
                    : [];
                for (let i = 0; i < concepts.length; i++) {
                    const concept = concepts[i] || {};
                    const rawValue = typeof concept.importance === "number"
                        ? concept.importance
                        : 0;
                    const conceptId = concept.id !== undefined && concept.id !== null
                        ? concept.id
                        : `${classId}-${i}`;
                    const key = this.conceptsAreClasswise
                        ? `${classId}:${conceptId}`
                        : String(conceptId);
                    let entry = map.get(key);
                    if (!entry) {
                        entry = {
                            id: conceptId,
                            label: concept.label,
                            values: new Map(),
                            maxAbs: 0,
                        };
                        map.set(key, entry);
                    }
                    if (entry.label === undefined || entry.label === null) {
                        entry.label = concept.label;
                    }
                    entry.values.set(classId, rawValue);
                    const absValue = Math.abs(rawValue);
                    if (absValue > entry.maxAbs) {
                        entry.maxAbs = absValue;
                    }
                }
            }

            const groups = [];
            map.forEach((entry) => {
                const values = [];
                let totalAbs = 0;
                for (const classId of classIds) {
                    if (!entry.values.has(classId)) {
                        if (this.conceptsAreClasswise) {
                            continue;
                        }
                        values.push({
                            classId,
                            rawValue: 0,
                            absValue: 0,
                        });
                        continue;
                    }
                    const rawValue = entry.values.get(classId);
                    const absValue = Math.abs(rawValue);
                    totalAbs += absValue;
                    values.push({
                        classId,
                        rawValue,
                        absValue,
                    });
                }
                if (!values.length) {
                    return;
                }
                groups.push({
                    id: entry.id,
                    label: entry.label,
                    values,
                    maxAbs: entry.maxAbs,
                    totalAbs,
                });
            });

            groups.sort((a, b) => b.totalAbs - a.totalAbs);
            return groups;
        }

        _renderScale(maxValue) {
            const scaleContainer = document.getElementById(this.uniqueIdConceptsScale);
            if (!scaleContainer) return;

            scaleContainer.innerHTML = "";
            if (!Number.isFinite(maxValue) || maxValue <= 0) {
                return;
            }

            const scale = document.createElement("div");
            scale.classList.add("concept-barplot-scale");

            const line = document.createElement("div");
            line.classList.add("concept-barplot-scale-line");
            scale.appendChild(line);

            const { step, decimals } = this._getScaleStep(maxValue);
            if (step > 0) {
                const tickValues = new Set([-maxValue, 0, maxValue]);
                for (let value = step; value < maxValue; value += step) {
                    tickValues.add(value);
                    tickValues.add(-value);
                }

                const sortedTickValues = Array.from(tickValues).sort((a, b) => a - b);
                for (const value of sortedTickValues) {
                    const tick = document.createElement("div");
                    tick.classList.add("concept-barplot-scale-tick");
                    if (Math.abs(value) < 1e-12) {
                        tick.classList.add("is-zero");
                    }
                    tick.style.left = `${this._toScalePercent(value, maxValue)}%`;

                    const mark = document.createElement("div");
                    mark.classList.add("concept-barplot-scale-mark");
                    tick.appendChild(mark);

                    const label = document.createElement("div");
                    label.classList.add("concept-barplot-scale-label");
                    label.textContent = this._formatTickValue(value, decimals);
                    tick.appendChild(label);

                    scale.appendChild(tick);
                }
            }

            scaleContainer.appendChild(scale);
        }

        _getScaleStep(maxValue) {
            const steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10];
            let step = steps[0];

            for (const candidate of steps) {
                const tickCountPerSide = Math.floor(maxValue / candidate);
                if (tickCountPerSide >= 2 && tickCountPerSide <= 4) {
                    step = candidate;
                    break;
                }
                step = candidate;
            }

            if (maxValue < step) {
                step = maxValue;
            }

            const decimals = Math.min(
                4,
                Math.max(this._countDecimals(step), this._countDecimals(maxValue))
            );

            return { step, decimals };
        }

        _countDecimals(value) {
            if (!Number.isFinite(value)) {
                return 0;
            }

            const text = value.toString().toLowerCase();
            if (text.includes("e-")) {
                const [base, exponentText] = text.split("e-");
                const exponent = parseInt(exponentText, 10);
                const fraction = base.includes(".") ? base.split(".")[1].length : 0;
                return exponent + fraction;
            }

            if (!text.includes(".")) {
                return 0;
            }
            return text.split(".")[1].length;
        }

        _toScalePercent(value, maxValue) {
            if (!Number.isFinite(maxValue) || maxValue <= 0) {
                return 50;
            }
            const ratio = (value + maxValue) / (2 * maxValue);
            return Math.max(0, Math.min(ratio, 1)) * 100;
        }

        _formatTickValue(value, decimals) {
            if (!Number.isFinite(value)) {
                return "";
            }
            const fixed = value.toFixed(decimals);
            const normalized = fixed.replace(/\.?0+$/, "");
            return normalized === "-0" ? "0" : normalized;
        }

        _formatLabel(label, conceptId) {
            if (Array.isArray(label)) {
                return label.join("\n");
            }
            if (label === undefined || label === null) {
                return `Concept #${conceptId}`;
            }
            return String(label);
        }
    };
})();
