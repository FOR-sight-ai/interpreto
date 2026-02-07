(function () {
    /**
     * StyleComputer - Pure functions for computing styles
     */
    window.StyleComputer = {
        /**
         * Convert a hex color string to an RGB array
         * @param {string} hex - The hex color string (e.g. "#ff0000")
         * @returns {number[]} An array of RGB values [r, g, b]
         */
        hexToRgb(hex) {
            hex = hex.replace("#", "");
            const r = parseInt(hex.substring(0, 2), 16);
            const g = parseInt(hex.substring(2, 4), 16);
            const b = parseInt(hex.substring(4, 6), 16);
            return [r, g, b];
        },

        /**
         * Resolve a color from a colormap object
         * @param {object|null} colormap - Object mapping ids to colors
         * @param {number|string} id - Id to resolve
         * @returns {string|null} Color string or null if missing
         */
        getColorFromMap(colormap, id) {
            if (!colormap) {
                return null;
            }
            const key = String(id);
            return colormap[key] || colormap[id] || null;
        },

        /**
         * Check whether a color is in the tab10 palette
         * @param {string|null} colorHex - Hex color string
         * @returns {boolean}
         */
        isTab10Color(colorHex) {
            if (!colorHex) {
                return false;
            }
            const normalized = colorHex.toLowerCase();
            const withHash = normalized.startsWith('#') ? normalized : `#${normalized}`;
            return [
                '#1f77b4',
                '#ff7f0e',
                '#2ca02c',
                '#d62728',
                '#9467bd',
                '#8c564b',
                '#e377c2',
                '#7f7f7f',
                '#bcbd22',
                '#17becf',
            ].includes(withHash);
        },


        /**
         * Normalize an alpha value based on min/max
         * @param {number} alpha - The raw alpha value
         * @param {number} min - The minimum value (for negative normalization)
         * @param {number} max - The maximum value (for positive normalization)
         * @returns {number} Normalized alpha between -1 and 1
         */
        normalizeAlpha(alpha, min, max) {
            if (alpha < 0) {
                return -(alpha / min);
            }
            return alpha / max;
        },

        /**
         * Compute the CSS style for a word based on attribution value
         * @param {number} alpha - The attribution value
         * @param {object} classMeta - Class metadata { positive_color, negative_color, min, max }
         * @param {object} options - Style options { normalize: boolean, highlightBorder: boolean }
         * @returns {string} A CSS style string
         */
        computeWordStyle(alpha, classMeta, options = {}) {
            const { normalize = true, highlightBorder = false } = options;

            let normalizedAlpha = alpha;
            if (normalize) {
                normalizedAlpha = this.normalizeAlpha(alpha, classMeta.min, classMeta.max);
            }

            // Select color based on sign
            const colorHex = normalizedAlpha < 0 ? classMeta.negative_color : classMeta.positive_color;
            const color = this.hexToRgb(colorHex);

            const absAlpha = Math.abs(normalizedAlpha);
            const alphaRatio = highlightBorder ? 0.5 : 1.0;

            const borderColor = [...color, absAlpha];
            const backgroundColor = [...color, absAlpha * alphaRatio];

            // Compute perceived brightness for text color
            const brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2];
            const effectiveAlpha = absAlpha * alphaRatio;

            let style = `background-color: rgba(${backgroundColor.join(",")});`;

            if (highlightBorder) {
                style += `outline-color: rgba(${borderColor.join(",")});`;
            } else {
                style += "outline-color: transparent;";
            }

            // Switch to white text for dark backgrounds
            if (effectiveAlpha >= 0.35 && brightness < 150) {
                style += "color: white;";
            }

            return style;
        },

        /**
         * Build custom style string from style object
         * @param {object|null} customStyle - Custom style object { property: value }
         * @returns {string} CSS style string
         */
        buildCustomStyle(customStyle) {
            if (!customStyle) return "";

            let style = "";
            for (const [key, value] of Object.entries(customStyle)) {
                style += `${key}: ${value};`;
            }
            return style;
        },

        /**
         * Format a tooltip value
         * @param {number} alpha - The attribution value
         * @param {number} precision - Decimal precision
         * @returns {string} Formatted tooltip text
         */
        formatTooltip(alpha, precision = 3) {
            return alpha.toFixed(precision);
        },

        /**
         * Get readable text color for a background color
         * @param {number[]|null} rgb - RGB array
         * @returns {string} CSS color value
         */
        getReadableTextColor(rgb) {
            if (
                !Array.isArray(rgb) ||
                rgb.length < 3 ||
                rgb.some((value) => !Number.isFinite(value))
            ) {
                return "#fff";
            }
            const brightness = this.getBrightness(rgb);
            return brightness < 150 ? "#fff" : "#000";
        },

        /**
         * Build label style for selectable items (classes/concepts)
         * @param {string|null} baseColor - Default color for the label
         * @param {boolean} isActive - Whether the label is hovered
         * @param {boolean} isSelected - Whether the label is selected
         * @param {object} options - { onClickColorMap, enableHighlight, showDefaultBackground, backgroundRgb }
         * @returns {string} CSS style string
         */
        buildLabelStyle(baseColor, isActive, isSelected, options = {}) {
            const {
                onClickColorMap = null,
                enableHighlight = true,
                showDefaultBackground = true,
                backgroundRgb = null,
            } = options;

            const shouldHighlight = enableHighlight && (isActive || isSelected);
            const showBackground = showDefaultBackground && baseColor && !shouldHighlight;

            let outlineColor = "transparent";
            if (shouldHighlight) {
                if (Array.isArray(onClickColorMap) && onClickColorMap.length >= 2) {
                    outlineColor = isSelected ? onClickColorMap[0] : onClickColorMap[1];
                } else if (baseColor) {
                    outlineColor = baseColor;
                }
            }

            let textColor;
            if (showBackground && baseColor && this.isTab10Color(baseColor)) {
                textColor = '#fff';
            } else {
                const textRgb = showBackground && baseColor
                    ? this.hexToRgb(baseColor)
                    : (backgroundRgb || this.getBackgroundRgb());
                textColor = this.getReadableTextColor(textRgb);
            }

            return (
                `background-color: ${showBackground ? baseColor : "transparent"};` +
                "text-shadow: none;" +
                `color: ${textColor};` +
                `outline-color: ${outlineColor};`
            );
        },

        /**
         * Find the dominant class for a token (class with highest positive attribution)
         * @param {Array} attributionsPerClass - Array of attribution values for each class
         * @returns {object} { classId: number|null, value: number|null } - null if no positive values
         */
        findDominantClass(attributionsPerClass) {
            let maxValue = -Infinity;
            let dominantClassId = null;

            for (let classId = 0; classId < attributionsPerClass.length; classId++) {
                const value = attributionsPerClass[classId];
                if (value > 0 && value > maxValue) {
                    maxValue = value;
                    dominantClassId = classId;
                }
            }

            if (dominantClassId === null) {
                return { classId: null, value: null };
            }

            return { classId: dominantClassId, value: maxValue };
        },

        /**
         * Compute the CSS style for a word in default multi-class view
         * @param {number} alpha - The attribution value (already the max positive)
         * @param {string} colorHex - The class color
         * @param {number} max - The max value for normalization
         * @param {object} options - Style options { highlightBorder: boolean }
         * @returns {string} A CSS style string
         */
        computeDefaultClassStyle(alpha, colorHex, max, options = {}) {
            const { highlightBorder = false } = options;

            // Normalize to 0-1 range (alpha is already positive)
            const normalizedAlpha = max > 0 ? alpha / max : 0;
            const color = this.hexToRgb(colorHex);

            const alphaRatio = highlightBorder ? 0.5 : 1.0;
            const borderColor = [...color, normalizedAlpha];
            const backgroundColor = [...color, normalizedAlpha * alphaRatio];

            // Compute perceived brightness for text color
            const brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2];
            const effectiveAlpha = normalizedAlpha * alphaRatio;
            const isTab10 = this.isTab10Color(colorHex);

            let style = `background-color: rgba(${backgroundColor.join(",")});`;

            if (highlightBorder) {
                style += `outline-color: rgba(${borderColor.join(",")});`;
            } else {
                style += "outline-color: transparent;";
            }

            // Switch to white text for dark backgrounds
            if (effectiveAlpha >= 0.35 && (brightness < 150 || isTab10)) {
                style += "color: white;";
            }

            return style;
        },

        /**
         * Parse an rgb/rgba CSS string into an RGB array
         * @param {string} value - CSS rgb/rgba string
         * @returns {number[]|null} RGB array or null if not parsed
         */
        parseRgb(value) {
            const match = value.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/i);
            if (!match) {
                return null;
            }
            return [
                parseInt(match[1], 10),
                parseInt(match[2], 10),
                parseInt(match[3], 10),
            ];
        },

        /**
         * Read the computed background color for an element
         * @param {HTMLElement} element - Element to read from
         * @param {number[]} fallback - Fallback RGB value
         * @returns {number[]} RGB array
         */
        getBackgroundRgb(element = document.body, fallback = [255, 255, 255]) {
            const rgb = this.parseRgb(window.getComputedStyle(element).backgroundColor);
            return rgb || fallback;
        },

        /**
         * Mix two RGB colors based on a ratio
         * @param {number[]} background - Background RGB
         * @param {number[]} foreground - Foreground RGB
         * @param {number} ratio - Ratio in [0, 1]
         * @returns {number[]} Mixed RGB
         */
        mixColors(background, foreground, ratio) {
            const mix = [];
            for (let i = 0; i < 3; i++) {
                const value = Math.round(background[i] + (foreground[i] - background[i]) * ratio);
                mix.push(Math.max(0, Math.min(255, value)));
            }
            return mix;
        },

        /**
         * Compute perceived brightness for readability decisions
         * @param {number[]} rgb - RGB array
         * @returns {number} Brightness value
         */
        getBrightness(rgb) {
            return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
        },

        /**
         * Build a readable style for a solid background color
         * @param {number[]} rgb - RGB array
         * @returns {string} CSS style string
         */
        buildReadableStyle(rgb) {
            const brightness = this.getBrightness(rgb);
            let style = `background-color: rgb(${rgb.join(",")});`;
            style += `outline-color: rgb(${rgb.join(",")});`;
            if (brightness < 150) {
                style += "color: white;";
            }
            return style;
        },

        /**
         * Compute the CSS style for a concept value using a mixed background
         * @param {number} value - Raw concept value
         * @param {number} maxValue - Max value for normalization
         * @param {string} colorHex - Base color
         * @param {number[]} backgroundRgb - Background color
         * @returns {string} CSS style string
         */
        computeConceptStyle(value, maxValue, colorHex, backgroundRgb) {
            const baseColor = this.hexToRgb(colorHex);
            const clampedValue = Math.max(0, Math.min(value, maxValue));
            const ratio = maxValue > 0 ? clampedValue / maxValue : 0;
            const mixed = this.mixColors(backgroundRgb, baseColor, ratio);
            return this.buildReadableStyle(mixed);
        },

        /**
         * Compute the CSS style for a solid concept color
         * @param {string} colorHex - Base color
         * @returns {string} CSS style string
         */
        computeSolidStyle(colorHex) {
            const baseColor = this.hexToRgb(colorHex);
            return this.buildReadableStyle(baseColor);
        },
    };
})();
