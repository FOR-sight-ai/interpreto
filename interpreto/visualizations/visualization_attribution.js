(function () {
  /**
   * DataVisualizationAttribution class (IIFE)
   * @param {string} uniqueIdInputs - The unique id of the div containing the inputs
   * @param {string} uniqueIdOutputs - The unique id of the div containing the outputs
   * @param {boolean} highlightBorder - Wether to highlight the border of the words or not
   * @param {string} jsonData - The JSON data containing the classes, inputs and outputs
   */
  window.DataVisualizationAttribution = class DataVisualizationAttribution {
    static DisplayType = {
      SINGLE_CLASS_ATTRIBUTION: 0, // simple attribution, only display the attribution for the input words
      MULTI_CLASS_ATTRIBUTION: 1, // multi-class attribution, display the attributions for the input words per class
      GENERATION_ATTRIBUTION: 2, // generation attribution, display the attributions for the input words per output
    };

    constructor(
      displayType,
      uniqueIdClasses,
      uniqueIdInputs,
      uniqueIdOutputs,
      highlightBorder,
      jsonData
    ) {
      console.log("Creating DataVisualizationAttribution");
      console.log("uniqueIdClasses: " + uniqueIdClasses);
      console.log("uniqueIdInputs: " + uniqueIdInputs);
      console.log("uniqueIdOutputs: " + uniqueIdOutputs);
      this.uniqueIdClasses = uniqueIdClasses;
      this.uniqueIdInputs = uniqueIdInputs;
      this.uniqueIdOutputs = uniqueIdOutputs;
      this.activatedClassId = null;
      this.selectedClassId = null;
      this.currentOutputId = null;
      this.selectedOutputId = null;
      this.highlightBorder = highlightBorder === "True";
      this.jsonData = JSON.parse(jsonData);
      this.displayType = displayType;
      // Classes, Inputs, Outputs creation (style is applied when selecting different elements)
      //                inputs    outputs   class
      // single class    many         0         1
      // multi class     many         0      many
      // generation      many      many         0
      switch (this.displayType) {
        case DataVisualizationAttribution.DisplayType.GENERATION_ATTRIBUTION:
          console.log("GENERATION_ATTRIBUTION");
          this.currentOutputId = null;
          this.activatedClassId = 0;
          this.selectedClassId = 0;
          this.createInputs();
          this.createOutputs();
          break;
        case DataVisualizationAttribution.DisplayType.MULTI_CLASS_ATTRIBUTION:
          console.log("MULTI_CLASS_ATTRIBUTION");
          this.currentOutputId = 0;
          this.activatedClassId = null;
          this.createClasses();
          this.createInputs();
          break;
        case DataVisualizationAttribution.DisplayType.SINGLE_CLASS_ATTRIBUTION:
        default:
          console.log("SINGLE_CLASS_ATTRIBUTION");
          // Select by default the only class available
          this.currentOutputId = 0;
          this.activatedClassId = 0;
          this.createClass();
          this.createInputs();
      }
      this.refreshInputsStyles();
      this.refreshOutputsStyles();
    }

    /**
     * Convert a hex color string to an RGB array
     * @param {string} hex - The hex color string (e.g. "#ff0000")
     * @returns {number[]} An array of RGB values [r, g, b]
     */
    hexToRgb(hex) {
      hex = hex.replace('#', '');
      let r = parseInt(hex.substring(0, 2), 16);
      let g = parseInt(hex.substring(2, 4), 16);
      let b = parseInt(hex.substring(4, 6), 16);
      return [r, g, b];
    }

    /**
     * Generate the CSS style for a word, depending on the class attribution
     *
     * @param {number} alpha The attribution value (between 0 and 1)
     * @param {number} min Min value for the class attribution (for normalization)
     * @param {number} max Max value for the class attribution (for normalization)
     * @param {number} classId The current class selected
     * @param {boolean} normalize Wether to normalize the alpha value with min and max
     * @returns {string} A CSS style string
     */
    getStyleForWord(alpha, min, max, classId, normalize) {
      let negativeColor = this.jsonData.classes[classId].negative_color;
      let positiveColor = this.jsonData.classes[classId].positive_color;

      if (normalize) {
        if (alpha < 0) {
          alpha = - (alpha / min);
        } else {
          alpha = alpha / max;
        }
      }
      
      // Compute the color of the word based on the alpha value     
      let color = alpha < 0 ? negativeColor : positiveColor;
      // convert str color to array of numbers
      color = this.hexToRgb(color);
      
      const alphaRatio = this.highlightBorder ? 0.5 : 1.0;
      const borderColor = [...color, Math.abs(alpha)];
      const backgroundColor = [...color, Math.abs(alpha) * alphaRatio]; // we actually dim the inside of the word to highlight the border

      var style = `background-color: rgba(${backgroundColor.join(",")});`;
      if (this.highlightBorder === true) {
        style += `outline-color: rgba(${borderColor.join(",")});`;
      } else {
        style += "outline-color: transparent;";
      }
      return style;
    }

    /**
     * Create the class element in the DOM, attached to the uniqueIdClasses div element (mono class attribution)
     */
    createClass() {
      // display the list of classes in 'unique_id_classes'
      var mainClassesDiv = document.getElementById(this.uniqueIdClasses);

      console.log("Creating " + this.jsonData.classes.length + " class of name " + this.jsonData.classes[0].name);
      // Add label for the class
      var currentClass = this.jsonData.classes[0];
      var classElement = document.createElement("div");
      classElement.classList.add("common-word-style");
      classElement.classList.add("class-style");
      classElement.textContent = currentClass.name;
      classElement.classId = 0;
      mainClassesDiv.appendChild(classElement);
    }

    /**
     * Create the classes buttons in the DOM, attached to the uniqueIdClasses div element (multi class attribution)
     */
    createClasses() {
      // display the list of classes in 'unique_id_classes'
      var mainClassesDiv = document.getElementById(this.uniqueIdClasses);

      console.log("Creating " + this.jsonData.classes.length + " class");
      // Add buttons for the classes
      for (let i = 0; i < this.jsonData.classes.length; i++) {
        var currentClass = this.jsonData.classes[i];
        var classElement = document.createElement("button");
        classElement.classList.add("common-word-style");
        classElement.classList.add("highlighted-word-style");
        classElement.classList.add("reactive-word-style");
        classElement.classList.add("class-style");
        classElement.onclick = function () {
          this.selectClass(i);
        }.bind(this);
        classElement.onmouseover = function () {
          this.activateClass(i);
        }.bind(this);
        classElement.onmouseout = function () {
          this.deactivateClass(i);
        }.bind(this);
        classElement.textContent = currentClass.name;
        classElement.classId = i;
        mainClassesDiv.appendChild(classElement);
      }
    }

    /**
     * Refresh the styles of the classes according to 'currentOutputId'
     */
    refreshClassesStyles() {
      // find the current ouput's classes and order them by value and
      // then display them in the correct order in the classes div
      var mainClassesDiv = document.getElementById(this.uniqueIdClasses);
      if (!mainClassesDiv) {
        return;
      }
      var classElements = mainClassesDiv.children;

      // Set the selected style to the selected class (and deselect the current class if no class is selected)
      console.log(
        "Refreshing the selected class with activatedClassId:",
        this.activatedClassId
      );
      for (let i = 0; i < classElements.length; i++) {
        var classElement = classElements[i];
        classElement.classList.toggle(
          "selected-style",
          classElement.classId === this.activatedClassId
        );
      }
    }

    /**
     * Activate a class by its id, this method is called when the mouse is over a class.
     * Refresh the styles of the inputs and outputs according to the selected class.
     *
     * @param {number} classId - The id of the class to activate
     */
    activateClass(classId) {
      console.log("Activating class " + classId);
      this.traceIds("before activateClass");
      this.activatedClassId = classId;
      this.refreshClassesStyles();
      this.refreshInputsStyles();
      this.refreshOutputsStyles();
    }

    /**
     * Deactivate a class by its id, this method is called when the mouse is not over a class anymore.
     * If a class was previously selected, it is reactivated.
     *
     * @param {number} classId - The id of the class to deactivate
     */
    deactivateClass(classId) {
      console.log("Deactivating class " + classId);
      this.traceIds("before deactivateClass");
      this.activateClass(this.selectedClassId);
    }

    /**
     * Select a class by its id: this method is called when the user clicks on a class.
     * If the class is already selected, it is deselected.
     *
     * @param {number} classId - The id of the class to select
     */
    selectClass(classId) {
      console.log("Selecting class " + classId);
      this.traceIds("before selectClass");

      if (this.selectedClassId === classId) {
        console.log("Class already selected, delecting it");
        this.selectedClassId = null;
        return;
      }
      this.selectedClassId = classId;
      this.activateClass(classId);
    }

    /**
     * Create the inputs div elements in the DOM, attached to the uniqueIdInputs div element
     * Each input sentence is displayed in a div element with a 'line-style' class
     * Each word in the sentence is displayed in a div element with a 'highlighted-word-style' class
     *
     */

    createInputs() {
      // display the list of input words in 'unique_id_inputs'
      // Create a div 'line-style' for each sentence
      var mainInputsDiv = document.getElementById(this.uniqueIdInputs);
      console.log("Creating input sentence:", this.jsonData.inputs.words);
      var sentenceElement = document.createElement("div");
      sentenceElement.classList.add("line-style");
      for (let j = 0; j < this.jsonData.inputs.words.length; j++) {
        var word = this.jsonData.inputs.words[j];
        word = this.normalizeSpecialChars(word);
        var wordElement = document.createElement("div");
        wordElement.classList.add("common-word-style");
        wordElement.classList.add("highlighted-word-style");
        wordElement.textContent = word;
        sentenceElement.appendChild(wordElement);
      }
      mainInputsDiv.appendChild(sentenceElement);
    }

    /**
     * Refresh the styles of the inputs according to the current class selected
     * and the current output selected
     * The style of each word is changed based on the class attribution
     * This value is displayed in a tooltip
     *
     */
    refreshInputsStyles() {
      console.log(
        "refreshInputsStyles(), activatedClassId: ",
        this.activatedClassId
      );

      var mainInputsDiv = document.getElementById(this.uniqueIdInputs);
      var sentenceElements = mainInputsDiv.children;

      // Get potential custom style from the user
      let customStyle = "";
      if (this.jsonData.custom_style) {
        console.log("\t refreshInputsStyles() custom style: ", this.jsonData.custom_style);
        for (const [key, value] of Object.entries(this.jsonData.custom_style)) {
          customStyle += `${key}: ${value};`;
        }
      }

      // iterate on sentences
      for (let i = 0; i < sentenceElements.length; i++) {
        var sentenceElement = sentenceElements[i];
        var wordElements = sentenceElement.children;

        // iterate on words & compute its alpha value based on the attributions
        for (let j = 0; j < wordElements.length; j++) {
          var wordElement = wordElements[j];
          if (this.activatedClassId == null || this.currentOutputId == null) {
            // Reset the style
            wordElement.style = customStyle;

            // Remove the tooltip if existing
            var previousTooltip = wordElement.getElementsByClassName("tooltiptext");
            if (previousTooltip.length > 0) {
              previousTooltip[0].remove();
            }
          } else {
            let alpha = this.jsonData.inputs.attributions[this.currentOutputId][j][this.activatedClassId];
            let minValue = this.jsonData.classes[this.activatedClassId].min;
            let maxValue = this.jsonData.classes[this.activatedClassId].max;

            // Generate the style for the word according to the alpha value
            let style = this.getStyleForWord(
              alpha,
              minValue,
              maxValue,
              this.activatedClassId,
              true
            );
            wordElement.style = style + customStyle;

            // Tooltip:
            // - Remove the previous tooltip if existing
            // - Add the new tooltip with the current class value
            var previousTooltip =
              wordElement.getElementsByClassName("tooltiptext");
            if (previousTooltip.length > 0) {
              previousTooltip[0].remove();
            }
            if (this.activatedClassId != null && this.currentOutputId != null) {
              var tooltip = document.createElement("span");
              tooltip.classList.add("tooltiptext");
              tooltip.textContent = alpha.toFixed(3);
              wordElement.appendChild(tooltip);
            }
          }
        }
      }
    }

    /**
     * Display some logs about the current selected class and output
     * @param {string} prefix a prefix to display in the console
     */
    traceIds(prefix) {
      console.log(
        "\t[" +
          prefix +
          "]" +
          "\tclass selected:" +
          this.selectedClassId +
          "/activated:" +
          this.activatedClassId +
          "\toutput selected:" +
          this.selectedOutputId +
          "/current:" +
          this.currentOutputId
      );
    }

    /**
     * Activate an output by its id, when the mouse is over it
     * Refresh the styles of the outputs according to the selected output
     * If a class was previously selected, it is deactivated
     *
     * @param {number} outputId - The id of the output to activate
     */
    activateOutput(outputId) {
      console.log("activateOutput(outputId:" + outputId + ")");
      this.traceIds("activateOutput");
      if (this.selectedOutputId != null && this.selectedClassId != null) {
        console.log(
          "\tA class and output is selected, not activating the output"
        );
        return;
      }

      this.currentOutputId = outputId;
      // When changing output, we reset the selected class
      if (
        this.displayType ===
        DataVisualizationAttribution.DisplayType.GENERATION_ATTRIBUTION
      ) {
        // GENERATION_ATTRIBUTION -> 1 class
        this.activatedClassId = 0;
        this.selectedClassId = null;
      } else {
        // CLASSES
        this.activatedClassId = null;
        this.selectedClassId = null;
      }

      this.traceIds("activateOutput");
      this.refreshOutputsStyles();
      this.refreshClassesStyles();
      this.refreshInputsStyles();
    }

    /**
     * Deactivate an output by its id, when the mouse is not over it anymore
     *
     * @param {number} outputId - The id of the output to deactivate
     */
    deactivateOutput(outputId) {
      console.log("deactivateOutput(outputId:" + outputId + ")");
      this.traceIds("deactivateOutput");

      if (this.selectedClassId != null) {
        console.log("\tA class is selected, not deactivating the output");
        return;
      }
      // Reactivating the current selected class
      if (this.selectedOutputId === outputId) {
        console.log("\tOutput already selected");
        return;
      }
      console.log(
        "\tReactivation of the saved selectedOutputId: " +
          this.selectedOutputId +
          " and classId: " +
          this.selectedClassId
      );
      this.activateOutput(this.selectedOutputId);
    }

    /**
     * Select an output by its id: this method is called when the
     * user clicks on an output to fix it
     *
     * @param {number} outputId - The id of the output to select
     */
    selectOutput(outputId) {
      console.log("selectOutput(outputId:" + outputId + ")");
      this.traceIds("selectOutput");

      if (this.selectedOutputId === outputId) {
        console.log("\tOutput already selected, deselecting it");
        this.selectedOutputId = null;
      } else {
        console.log("\tOutput selected: " + outputId);
        this.selectedOutputId = outputId;
      }

      // Reset the current class when the output is changed
      this.selectedClassId = null;
      this.traceIds("selectOutput");
      this.activateOutput(this.selectedOutputId);

      // In case of a generation attribution display, we select the class 0
      // by default (the null step is needed to refresh the display)
      if (
        this.displayType ===
        DataVisualizationAttribution.DisplayType.GENERATION_ATTRIBUTION
      ) {
        this.selectedClassId = 0;
        this.traceIds("selectOutput");
        this.activateOutput(this.selectedOutputId);
      }
    }

    /**
     * Normalize special characters in a word
     * This is useful to display special characters in the output
     * (e.g. \n, \t, etc.)
     * Replace special characters by adding a backslash to make them visible.
     * @param {string} word - The word to normalize
     * @returns {string} The normalized word
     *
     */
    normalizeSpecialChars(word) {
      let normalizedWord = word.replace(/\n/g, '\\n')
      .replace(/\r/g, '\\r')
      .replace(/\t/g, '\\t');
      return normalizedWord;
    }

    /**
     * Create the output div elements in the DOM, attached to the uniqueIdOutputs div element
     * Each output word is displayed in a button element with a 'highlighted-word-style' class
     * The style of the output word is changed based on the value of the class
     * The value of the class is displayed in a tooltip
     *
     */
    createOutputs() {
      console.log("Creating outputs");
      var mainOutputsDiv = document.getElementById(this.uniqueIdOutputs);

      // for each output word, display the word
      for (let i = 0; i < this.jsonData.outputs.words.length; i++) {
        var word = this.jsonData.outputs.words[i];
        word = this.normalizeSpecialChars(word);
       
        console.log("Output " + i + ": " + word);
        var outputElement = document.createElement("button");
        outputElement.classList.add("common-word-style");
        outputElement.classList.add("highlighted-word-style");
        outputElement.classList.add("reactive-word-style");
        outputElement.onclick = function () {
          this.selectOutput(i);
        }.bind(this);
        outputElement.onmouseover = function () {
          this.activateOutput(i);
        }.bind(this);
        outputElement.onmouseout = function () {
          this.deactivateOutput(i);
        }.bind(this);
        outputElement.textContent = word;
        mainOutputsDiv.appendChild(outputElement);
      }
    }

    /**
     * Refresh the styles of the outputs according to the current class and output
     *
     * The style of each word is changed based on the value of the class
     * The value of the class is displayed in a tooltip
     */
    refreshOutputsStyles() {
      var mainOutputsDiv = document.getElementById(this.uniqueIdOutputs);
      if (!mainOutputsDiv) {
        return;
      }
      console.log(
        "refreshOutputsStyles(), selected output: ",
        this.selectedOutputId
      );

      // Get potential custom style from the user
      let customStyle = "";
      if (this.jsonData.custom_style) {
        console.log("\t refreshInputsStyles() custom style: ", this.jsonData.custom_style);
        for (const [key, value] of Object.entries(this.jsonData.custom_style)) {
          customStyle += `${key}: ${value};`;
        }
      }

      for (let i = 0; i < mainOutputsDiv.children.length; i++) {
        // Update the style of each output word, based on its position relative to the current selected output
        var child = mainOutputsDiv.children[i];
        child.classList.toggle(
          "highlighted-word-style",
          i < this.currentOutputId
        );
        child.classList.toggle("selected-style", i === this.currentOutputId);

        let alpha = 0.0;
        if (this.activatedClassId != null && i < this.currentOutputId) {
          alpha =
            this.jsonData.outputs.attributions[this.currentOutputId][i][
              this.activatedClassId
            ];
          let minValue = this.jsonData.classes[this.activatedClassId].min;
          let maxValue = this.jsonData.classes[this.activatedClassId].max;
          let style = this.getStyleForWord(
            alpha,
            minValue,
            maxValue,
            this.activatedClassId,
            true
          );
          child.style = style + customStyle;
        } else {
          child.style = customStyle; // reset the style
        }

        // Tooltip for the output word
        // - Remove the previous tooltip if existing
        // - Add the new tooltip with the current class value
        var previousTooltip = child.getElementsByClassName("tooltiptext");
        if (previousTooltip.length > 0) {
          previousTooltip[0].remove();
        }
        if (this.activatedClassId != null && i < this.currentOutputId) {
          var tooltip = document.createElement("span");
          tooltip.classList.add("tooltiptext");
          tooltip.textContent = alpha.toFixed(3);
          child.appendChild(tooltip);
        }
      }
    }
  };
})();
