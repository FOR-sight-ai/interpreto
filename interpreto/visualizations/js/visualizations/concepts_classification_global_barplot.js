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
            this.topK = Math.max(0, parseInt(this.data.top_k || 0, 10));

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
                this._renderScale(0, 0);
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

            let globalMin = Number.POSITIVE_INFINITY;
            let globalMax = Number.NEGATIVE_INFINITY;
            for (const group of groups) {
                for (const value of group.values) {
                    if (value.rawValue < globalMin) {
                        globalMin = value.rawValue;
                    }
                    if (value.rawValue > globalMax) {
                        globalMax = value.rawValue;
                    }
                }
            }

            if (!Number.isFinite(globalMin) || !Number.isFinite(globalMax)) {
                globalMin = 0;
                globalMax = 0;
            }

            this._renderScale(globalMin, globalMax);

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
                    const zeroPercent = this._toScalePercent(0, globalMin, globalMax);
                    const valuePercent = this._toScalePercent(value.rawValue, globalMin, globalMax);
                    const leftPercent = Math.min(zeroPercent, valuePercent);
                    const widthPercent = Math.abs(valuePercent - zeroPercent);

                    const track = document.createElement("div");
                    track.classList.add("concept-barplot-bar-track", "highlighted-word-style");
                    track.style.setProperty("--concept-barplot-zero-percent", `${zeroPercent}%`);

                    const fill = document.createElement("div");
                    fill.classList.add("concept-barplot-bar-fill");
                    fill.style.left = `${leftPercent}%`;
                    fill.style.width = `${widthPercent}%`;
                    if (value.rawValue < 0) {
                        fill.classList.add("is-negative");
                    } else {
                        fill.classList.add("is-positive");
                    }
                    fill.style.right = "auto";
                    fill.style.backgroundColor = classColor;
                    if (widthPercent > 0) {
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
                return Array.from(this.selectedClassIds);
            }
            if (this.hoveredClassId !== null) {
                return [this.hoveredClassId];
            }
            return [];
        }

        _buildConceptGroups(classIds) {
            const candidateKeys = new Set();
            const conceptMeta = new Map();
            const conceptValuesByClass = new Map();
            const firstClassOrder = new Map();
            const firstClassId = classIds[0];

            for (const classId of classIds) {
                const concepts = Array.isArray(this.data.concepts)
                    ? this.data.concepts[classId] || []
                    : [];
                const values = new Map();
                const topLimit = this.topK > 0
                    ? Math.min(this.topK, concepts.length)
                    : concepts.length;

                for (let i = 0; i < concepts.length; i++) {
                    const concept = concepts[i] || {};
                    const conceptId = concept.id !== undefined && concept.id !== null
                        ? concept.id
                        : `${classId}-${i}`;
                    const key = this.conceptsAreClasswise
                        ? `${classId}:${conceptId}`
                        : String(conceptId);
                    const rawValue = typeof concept.importance === "number"
                        ? concept.importance
                        : 0;

                    values.set(key, rawValue);
                    if (!conceptMeta.has(key)) {
                        conceptMeta.set(key, {
                            id: conceptId,
                            label: concept.label,
                        });
                    }

                    if (classId === firstClassId && !firstClassOrder.has(key)) {
                        firstClassOrder.set(key, i);
                    }

                    if (i < topLimit) {
                        candidateKeys.add(key);
                    }
                }

                conceptValuesByClass.set(classId, values);
            }

            const groups = [];
            for (const key of candidateKeys) {
                const meta = conceptMeta.get(key);
                if (!meta) {
                    continue;
                }

                const values = [];
                let totalAbs = 0;
                let maxAbs = 0;

                for (const classId of classIds) {
                    const classValues = conceptValuesByClass.get(classId);
                    const hasValue = classValues ? classValues.has(key) : false;

                    if (!hasValue && this.conceptsAreClasswise) {
                        continue;
                    }

                    const rawValue = hasValue
                        ? classValues.get(key)
                        : 0;
                    const absValue = Math.abs(rawValue);
                    totalAbs += absValue;
                    if (absValue > maxAbs) {
                        maxAbs = absValue;
                    }
                    values.push({
                        classId,
                        rawValue,
                        absValue,
                    });
                }

                if (!values.length) {
                    continue;
                }

                groups.push({
                    id: meta.id,
                    label: meta.label,
                    values,
                    maxAbs,
                    totalAbs,
                    firstClassRank: firstClassOrder.has(key)
                        ? firstClassOrder.get(key)
                        : Number.POSITIVE_INFINITY,
                });
            }

            groups.sort((a, b) => {
                if (a.firstClassRank !== b.firstClassRank) {
                    return a.firstClassRank - b.firstClassRank;
                }
                if (b.totalAbs !== a.totalAbs) {
                    return b.totalAbs - a.totalAbs;
                }
                return String(a.id).localeCompare(String(b.id));
            });
            return groups;
        }

        _renderScale(minValue, maxValue) {
            const scaleContainer = document.getElementById(this.uniqueIdConceptsScale);
            if (!scaleContainer) return;

            scaleContainer.innerHTML = "";
            if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
                return;
            }

            const scale = document.createElement("div");
            scale.classList.add("concept-barplot-scale");

            const line = document.createElement("div");
            line.classList.add("concept-barplot-scale-line");
            scale.appendChild(line);

            const { step, decimals } = this._getScaleStep(minValue, maxValue);
            if (step > 0) {
                const tickValues = new Set();
                const epsilon = step / 1000;
                const start = Math.ceil(minValue / step) * step;
                const end = maxValue - epsilon;

                for (let value = start; value <= end; value += step) {
                    const roundedValue = Number(value.toFixed(Math.max(6, decimals + 2)));
                    if (roundedValue <= minValue + epsilon || roundedValue >= maxValue - epsilon) {
                        continue;
                    }
                    tickValues.add(roundedValue);
                }

                if (minValue < 0 && maxValue > 0) {
                    tickValues.add(0);
                }

                const sortedTickValues = Array.from(tickValues).sort((a, b) => a - b);
                for (const value of sortedTickValues) {
                    const tick = document.createElement("div");
                    tick.classList.add("concept-barplot-scale-tick");
                    if (Math.abs(value) < 1e-12) {
                        tick.classList.add("is-zero");
                    }
                    tick.style.left = `${this._toScalePercent(value, minValue, maxValue)}%`;

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

        _getScaleStep(minValue, maxValue) {
            const steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10];
            const span = maxValue - minValue;
            if (!Number.isFinite(span) || span <= 0) {
                return { step: 0, decimals: 0 };
            }

            let step = steps[steps.length - 1];

            for (const candidate of steps) {
                const tickCount = span / candidate;
                if (tickCount <= 8) {
                    step = candidate;
                    break;
                }
            }

            if (span < step) {
                step = span;
            }

            const decimals = Math.min(
                4,
                Math.max(
                    this._countDecimals(step),
                    this._countDecimals(minValue),
                    this._countDecimals(maxValue)
                )
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

        _toScalePercent(value, minValue, maxValue) {
            if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
                return 50;
            }

            if (maxValue <= minValue) {
                if (maxValue <= 0) {
                    return 100;
                }
                if (minValue >= 0) {
                    return 0;
                }
                return 50;
            }

            const ratio = (value - minValue) / (maxValue - minValue);
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
