(function () {
  /**
   * DataVisualization class (IIFE)
   * @param {string} uniqueIdConcepts - The unique id of the div containing the concepts
   * @param {string} uniqueIdInputs - The unique id of the div containing the inputs
   * @param {string} uniqueIdOutputs - The unique id of the div containing the outputs
   * @param {number} topk - The number of top concepts to display
   * @param {boolean} highlightBorder - Wether to highlight the border of the words or not
   * @param {string} jsonData - The JSON data containing the concepts, inputs and outputs
   */
  window.DataVisualization = class DataVisualization {
    static DisplayType = {
      SINGLE_CLASS_ATTRIBUTION: 1, // simple attribution, only display the attribution for the input words
      MULTI_CLASS_ATTRIBUTION: 2, // multi-class attribution, display the attributions for the input words per class
      GENERATION_ATTRIBUTION: 3, // generation attribution, display the attributions for the input words per output
      CONCEPTS: 4, // encoder/decoder attribution with concepts, display the attributions for the input & output words, per concept, per output
    };

    constructor(
      uniqueIdConcepts,
      uniqueIdInputs,
      uniqueIdOutputs,
      topk,
      highlightBorder,
      jsonData
    ) {
      console.log("Creating DataVisualization");
      console.log("uniqueIdConcepts: " + uniqueIdConcepts);
      console.log("uniqueIdInputs: " + uniqueIdInputs);
      console.log("uniqueIdOutputs: " + uniqueIdOutputs);
      this.uniqueIdConcepts = uniqueIdConcepts;
      this.uniqueIdInputs = uniqueIdInputs;
      this.uniqueIdOutputs = uniqueIdOutputs;
      this.currentConceptId = null;
      this.selectedConceptId = null;
      this.currentOutputId = null;
      this.selectedOutputId = null;
      this.highlightBorder = highlightBorder === "True";
      this.jsonData = JSON.parse(jsonData);
      this.topk = topk;

      this.displayType = DataVisualization.DisplayType.SINGLE_CLASS_ATTRIBUTION;
      if (this.uniqueIdConcepts != null) {
        if (this.uniqueIdOutputs != null) {
          this.displayType = DataVisualization.DisplayType.CONCEPTS;
        } else {
          this.displayType =
            DataVisualization.DisplayType.MULTI_CLASS_ATTRIBUTION;
        }
      } else if (this.uniqueIdOutputs != null) {
        this.displayType = DataVisualization.DisplayType.GENERATION_ATTRIBUTION;
      }

      // Concepts, Inputs, Outputs creation (style is applied when selecting different elements)
      //                inputs    outputs   concepts/class
      // single class    many         0         1
      // multi class     many         0      many
      // generation      many      many         0
      // concepts        many      many      many
      switch (this.displayType) {
        case DataVisualization.DisplayType.CONCEPTS:
          console.log("CONCEPTS");
          this.currentOutputId = null;
          this.currentConceptId = null;
          this.createConcepts();
          this.createInputs();
          this.createOutputs();
          break;
        case DataVisualization.DisplayType.GENERATION_ATTRIBUTION:
          console.log("GENERATION_ATTRIBUTION");
          this.currentOutputId = null;
          this.currentConceptId = 0;
          this.selectedConceptId = 0;
          this.createInputs();
          this.createOutputs();
          break;
        case DataVisualization.DisplayType.MULTI_CLASS_ATTRIBUTION:
          console.log("MULTI_CLASS_ATTRIBUTION");
          this.currentOutputId = 0;
          this.currentConceptId = null;
          this.createConcepts();
          this.createInputs();
          break;
        case DataVisualization.DisplayType.SINGLE_CLASS_ATTRIBUTION:
        default:
          console.log("SINGLE_CLASS_ATTRIBUTION");
          // Select by default the only class available
          this.currentOutputId = 0;
          this.currentConceptId = 0;
          this.createInputs();
      }
      this.refreshInputsStyles();
      this.refreshOutputsStyles();
    }

    /**
     * Generate the CSS style for a word, depending on the concept attribution
     *
     * @param {number} alpha The attribution value (between 0 and 1)
     * @param {number} min Min value for the concept attribution (for normalization)
     * @param {number} max Max value for the concept attribution (for normalization)
     * @param {number} conceptId The current concept selected
     * @param {boolean} normalize Wether to normalize the alpha value with min and max
     * @returns {string} A CSS style string
     */
    getStyleForWord(alpha, min, max, conceptId, normalize) {
      let color = [0, 0, 0];
      if (conceptId != null) {
        const concept = this.jsonData.concepts[conceptId];
        color = concept.color.map((c) => Math.floor(c * 255));
      }
      if (normalize) {
        alpha = (alpha - min) / (max - min);
      }
      const alphaRatio = this.highlightBorder ? 0.75 : 1.0;
      const borderColor = [...color, alpha];
      const backgroundColor = [...color, alpha * alphaRatio]; // we actually dim the inside of the word to highlight the border
      var style = `background-color: rgba(${backgroundColor.join(",")});`;
      if (this.highlightBorder === true) {
        style += `border-color: rgba(${borderColor.join(",")});`;
      } else {
        style += "border-color: transparent;";
      }
      return style;
    }

    /**
     * Activate a concept by its id, this method is called when the mouse is over a concept.
     * Refresh the styles of the inputs and outputs according to the selected concept.
     *
     * @param {number} conceptId - The id of the concept to activate
     */
    activateConcept(conceptId) {
      console.log("Activating concept " + conceptId);
      this.traceIds();
      this.currentConceptId = conceptId;
      this.refreshConceptsStyles();
      this.refreshInputsStyles();
      this.refreshOutputsStyles();
    }

    /**
     * Deactivate a concept by its id, this method is called when the mouse is not over a concept anymore.
     * If a concept was previously selected, it is reactivated.
     *
     * @param {number} conceptId - The id of the concept to deactivate
     */
    deactivateConcept(conceptId) {
      console.log("Deactivating concept " + conceptId);
      this.traceIds();
      this.activateConcept(this.selectedConceptId);
    }

    /**
     * Select a concept by its id: this method is called when the user clicks on a concept.
     * If the concept is already selected, it is deselected.
     *
     * @param {number} conceptId - The id of the concept to select
     */
    selectConcept(conceptId) {
      console.log("Selecting concept " + conceptId);
      this.traceIds();

      if (this.selectedConceptId === conceptId) {
        console.log("Concept already selected, delecting it");
        this.selectedConceptId = null;
        return;
      }
      this.selectedConceptId = conceptId;
      this.activateConcept(conceptId);
    }

    /**
     * Create the concepts buttons in the DOM, attached to the uniqueIdConcepts div element
     */
    createConcepts() {
      // display the list of concepts in 'unique_id_concepts'
      var mainConceptsDiv = document.getElementById(this.uniqueIdConcepts);

      console.log("Creating " + this.jsonData.concepts.length + " concepts");
      // Add buttons for the concepts
      for (let i = 0; i < this.jsonData.concepts.length; i++) {
        var concept = this.jsonData.concepts[i];
        var conceptElement = document.createElement("button");
        conceptElement.classList.add("common-word-style");
        conceptElement.classList.add("highlighted-word-style");
        conceptElement.classList.add("reactive-word-style");
        conceptElement.classList.add("concept-style");
        // Use min/max of 0/1 to force a fully colored style for the concept buttons
        conceptElement.style = this.getStyleForWord(0.5, 0, 1, i, true, true);
        conceptElement.onclick = function () {
          this.selectConcept(i);
        }.bind(this);
        conceptElement.onmouseover = function () {
          this.activateConcept(i);
        }.bind(this);
        conceptElement.onmouseout = function () {
          this.deactivateConcept(i);
        }.bind(this);
        conceptElement.textContent = concept.name;
        conceptElement.conceptId = i;
        mainConceptsDiv.appendChild(conceptElement);
      }
    }

    /**
     * Refresh the styles of the concepts according to 'currentOutputId' and 'topk'
     */
    refreshConceptsStyles() {
      // find the current ouput's concepts and order them by value, take the topk
      // then display them in the correct order in the concepts div
      var mainConceptsDiv = document.getElementById(this.uniqueIdConcepts);
      if (!mainConceptsDiv) {
        return;
      }
      var conceptElements = mainConceptsDiv.children;

      // If no output is selected yet, display all the concepts
      if (this.currentOutputId == null) {
        for (let i = 0; i < conceptElements.length; i++) {
          var conceptElement = conceptElements[i];
          conceptElement.style.visibility = "visible";
        }
        return;
      }

      // topk filtering when we have outputs attributions
      if (this.jsonData.outputs.attributions) {
        // Get the topk concepts for the current output
        var topk_concepts = this.getTopkConcepts(this.currentOutputId);
        console.log(
          "Refreshing concepts for output " +
            this.currentOutputId +
            ", using topk concepts " +
            topk_concepts
        );

        // Reoder the elements in the conceptsElements, with the topk elements in
        // first position
        conceptElements = Array.prototype.slice.call(conceptElements);
        conceptElements.sort(function (a, b) {
          var a_value = topk_concepts.indexOf(a.conceptId);
          var b_value = topk_concepts.indexOf(b.conceptId);
          if (a_value === -1) return 1;
          if (b_value === -1) return -1;
          return a_value - b_value;
        });

        // Append the elements in the correct order and hide the ones that are not in the topk
        conceptElements.forEach(function (element) {
          mainConceptsDiv.appendChild(element);
        });
        for (let i = 0; i < conceptElements.length; i++) {
          var conceptElement = conceptElements[i];
          if (this.topk == null || i < this.topk) {
            conceptElement.style.visibility = "visible";
          } else {
            conceptElement.style.visibility = "hidden";
          }
        }
      }

      // Set the selected style to the selected concept (and deselect the current concept if no concept is selected)
      console.log(
        "Refreshing the selected concept with currentConceptId:",
        this.currentConceptId
      );
      for (let i = 0; i < conceptElements.length; i++) {
        var conceptElement = conceptElements[i];
        conceptElement.classList.toggle(
          "selected-style",
          conceptElement.conceptId === this.currentConceptId
        );
      }
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
      for (let i = 0; i < this.jsonData.inputs.length; i++) {
        // iterate on each input sentence
        var sentence = this.jsonData.inputs[i];
        console.log("Creating input sentence:", sentence);
        var sentenceElement = document.createElement("div");
        sentenceElement.classList.add("line-style");
        for (let j = 0; j < sentence.words.length; j++) {
          var word = sentence.words[j];
          var wordElement = document.createElement("div");
          wordElement.classList.add("common-word-style");
          wordElement.classList.add("highlighted-word-style");
          wordElement.textContent = word;
          sentenceElement.appendChild(wordElement);
        }
        mainInputsDiv.appendChild(sentenceElement);
      }
    }

    /**
     * Refresh the styles of the inputs according to the current concept selected
     * and the current output selected
     * The style of each word is changed based on the concept attribution
     * This value is displayed in a tooltip
     *
     */
    refreshInputsStyles() {
      console.log(
        "refreshInputsStyles(), currentConceptId: ",
        this.currentConceptId
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
          let alpha = 0.0;
          let minValue = 0.0;
          let maxValue = 1.0;
          if (this.currentOutputId != null && this.currentConceptId != null) {
            alpha =
              this.jsonData.inputs[i].attributions[this.currentOutputId][j][
                this.currentConceptId
              ];
            minValue = this.jsonData.concepts[this.currentConceptId].min;
            maxValue = this.jsonData.concepts[this.currentConceptId].max;
          }
          // Generate the style for the word according to the alpha value
          let style = this.getStyleForWord(
            alpha,
            minValue,
            maxValue,
            this.currentConceptId,
            true
          );
          wordElement.style = style + customStyle;

          // Tooltip:
          // - Remove the previous tooltip if existing
          // - Add the new tooltip with the current concept value
          var previousTooltip =
            wordElement.getElementsByClassName("tooltiptext");
          if (previousTooltip.length > 0) {
            previousTooltip[0].remove();
          }
          if (this.currentConceptId != null && this.currentOutputId != null) {
            var tooltip = document.createElement("span");
            tooltip.classList.add("tooltiptext");
            tooltip.textContent = alpha.toFixed(3);
            wordElement.appendChild(tooltip);
          }
        }
      }
    }

    /**
     * Display some logs about the current selected concept and output
     * @param {string} prefix a prefix to display in the console
     */
    traceIds(prefix) {
      console.log(
        "\t[" +
          prefix +
          "]" +
          "\tconcept selected:" +
          this.selectedConceptId +
          "/" +
          this.currentConceptId +
          "\toutput selected:" +
          this.selectedOutputId +
          "/" +
          this.currentOutputId
      );
    }

    /**
     * Activate an output by its id, when the mouse is over it
     * Refresh the styles of the outputs according to the selected output
     * If a concept was previously selected, it is deactivated
     *
     * @param {number} outputId - The id of the output to activate
     */
    activateOutput(outputId) {
      console.log("activateOutput(outputId:" + outputId + ")");
      this.traceIds("activateOutput");
      if (this.selectedOutputId != null && this.selectedConceptId != null) {
        console.log(
          "\tA concept and output is selected, not activating the output"
        );
        return;
      }

      this.currentOutputId = outputId;
      // When changing output, we reset the selected concept
      if (
        this.displayType ===
        DataVisualization.DisplayType.GENERATION_ATTRIBUTION
      ) {
        // GENERATION_ATTRIBUTION -> 1 concept
        this.currentConceptId = 0;
        this.selectedConceptId = null;
      } else {
        // CONCEPTS
        this.currentConceptId = null;
        this.selectedConceptId = null;
      }

      this.traceIds("activateOutput");
      this.refreshOutputsStyles();
      this.refreshConceptsStyles();
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

      if (this.selectedConceptId != null) {
        console.log("\tA concept is selected, not deactivating the output");
        return;
      }
      // Reactivating the current selected concept
      if (this.selectedOutputId === outputId) {
        console.log("\tOutput already selected");
        return;
      }
      console.log(
        "\tReactivation of the saved selectedOutputId: " +
          this.selectedOutputId +
          " and conceptId: " +
          this.selectedConceptId
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

      // Reset the current concept when the output is changed
      this.selectedConceptId = null;
      this.traceIds("selectOutput");
      this.activateOutput(this.selectedOutputId);

      // In case of a generation attribution display, we select the concept 0
      // by default (the null step is needed to refresh the display)
      if (
        this.displayType ===
        DataVisualization.DisplayType.GENERATION_ATTRIBUTION
      ) {
        this.selectedConceptId = 0;
        this.traceIds("selectOutput");
        this.activateOutput(this.selectedOutputId);
      }
    }

    /**
     * Get the topk concepts for an outputId
     * @param {number} outputId - The id of the output
     * @returns An array of topk concepts
     *
     */
    getTopkConcepts(outputId) {
      console.log("Getting topk concepts for output " + outputId);
      if (this.currentOutputId == null) {
        // impossible: we need an output word selected in order to compute the
        // the topk concepts for the outputId output
        return [];
      }
      var attributions =
        this.jsonData.outputs.attributions[this.currentOutputId][outputId];
      console.log("Attributions for output " + outputId + ":", attributions);
      var ordered_concepts_ids = attributions
        .map((_, i) => i)
        .sort((a, b) => attributions[b] - attributions[a]);
      var topk_concepts = ordered_concepts_ids.slice(0, this.topk);
      return topk_concepts;
    }

    /**
     * Create the output div elements in the DOM, attached to the uniqueIdOutputs div element
     * Each output word is displayed in a button element with a 'highlighted-word-style' class
     * The style of the output word is changed based on the value of the concept
     * The value of the concept is displayed in a tooltip
     *
     */
    createOutputs() {
      console.log("Creating outputs");
      var mainOutputsDiv = document.getElementById(this.uniqueIdOutputs);

      // for each output word, display the word
      for (let i = 0; i < this.jsonData.outputs.words.length; i++) {
        var word = this.jsonData.outputs.words[i];
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
     * Refresh the styles of the outputs according to the current concept and output
     *
     * The style of each word is changed based on the value of the concept
     * The value of the concept is displayed in a tooltip
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
        console.log("\t refreshOutputStyles() custom style: ", this.jsonData.custom_style);
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

        // TODO: merge with the styling done in refreshInputsStyles ?
        let alpha = 0.0;
        let minValue = 0.0;
        let maxValue = 1.0;
        if (this.currentConceptId != null && i < this.currentOutputId) {
          alpha =
            this.jsonData.outputs.attributions[this.currentOutputId][i][
              this.currentConceptId
            ];
          minValue = this.jsonData.concepts[this.currentConceptId].min;
          maxValue = this.jsonData.concepts[this.currentConceptId].max;
          let style = this.getStyleForWord(
            alpha,
            minValue,
            maxValue,
            this.currentConceptId,
            true
          );
          child.style = style + customStyle;
        } else {
          child.style = customStyle; // reset the style
        }

        // Tooltip for the output word
        // - Remove the previous tooltip if existing
        // - Add the new tooltip with the current concept value
        var previousTooltip = child.getElementsByClassName("tooltiptext");
        if (previousTooltip.length > 0) {
          previousTooltip[0].remove();
        }
        if (this.currentConceptId != null && i < this.currentOutputId) {
          var tooltip = document.createElement("span");
          tooltip.classList.add("tooltiptext");
          tooltip.textContent = alpha.toFixed(3);
          child.appendChild(tooltip);
        }
      }
    }
  };
})();
