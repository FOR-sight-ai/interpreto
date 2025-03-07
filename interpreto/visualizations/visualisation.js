(function () {
  /**
   * DataVisualisation class (IIFE)
   * @param {string} uniqueIdConcepts - The unique id of the div containing the concepts
   * @param {string} uniqueIdInputs - The unique id of the div containing the inputs
   * @param {string} uniqueIdOutputs - The unique id of the div containing the outputs
   * @param {number} topk - The number of top concepts to display
   * @param {string} jsonData - The JSON data containing the concepts, inputs and outputs
   */
  window.DataVisualisation = class DataVisualisation {
    constructor(
      uniqueIdConcepts,
      uniqueIdInputs,
      uniqueIdOutputs,
      topk,
      jsonData
    ) {
      console.log("Creating DataVisualisation");
      console.log("uniqueIdConcepts: " + uniqueIdConcepts);
      console.log("uniqueIdInputs: " + uniqueIdInputs);
      console.log("uniqueIdOutputs: " + uniqueIdOutputs);
      this.uniqueIdConcepts = uniqueIdConcepts;
      this.uniqueIdInputs = uniqueIdInputs;
      this.uniqueIdOutputs = uniqueIdOutputs;
      this.jsonData = JSON.parse(jsonData);
      this.current_concept_id = null;
      this.selected_concept_id = null;
      this.current_output_id = null;
      this.selected_output_id = null;
      this.topk = topk;

      class DisplayType {
        static SINGLE_CLASS = 1; // simple attribution, only display the attribution for the input words
        static MULTI_CLASS = 2; // multi-class attribution, display the attributions for the input words per class
        static CONCEPTS = 3; // encoder/decoder attribution, display the attributions for the input words per concept per output
      }

      var displayType = DisplayType.SINGLE_CLASS;
      if (this.uniqueIdOutputs != null) {
        displayType = DisplayType.CONCEPTS;
      } else if (this.uniqueIdConcepts != null) {
        displayType = DisplayType.MULTI_CLASS;
      }

      // Concepts, Inputs, Outputs creation (style is applied when selecting different elements)
      switch (displayType) {
        case DisplayType.CONCEPTS:
          this.current_output_id = null;
          this.current_concept_id = null;

          this.createConcepts();
          this.createInputs();
          this.createOutputs();
          break;
        case DisplayType.MULTI_CLASS:
          this.current_output_id = 0;
          this.current_concept_id = null;
          this.createConcepts();
          this.createInputs();
          break;
        case DisplayType.SINGLE_CLASS:
        default:
          // Select by default the only class available
          this.current_output_id = 0;
          this.current_concept_id = 0;
          this.createInputs();
      }

      this.refreshInputsStyles();
    }

    /**
     * Generate the CSS style for a word, depending on the concept attribution
     *
     * @param {number} alpha The attribution value (between 0 and 1)
     * @param {number} min Min value for the concept attribution (for normalization)
     * @param {number} max Max value for the concept attribution (for normalization)
     * @param {number} concept_id The current concept selected
     * @param {boolean} normalize Wether to normalize the alpha value with min and max
     * @param {boolean} highlight_border Wether to highlight the border of the word
     * @returns {string} A CSS style string
     */
    getStyleForWord(alpha, min, max, concept_id, normalize, highlight_border) {
      // console.log("getStyleForWord: alpha: " + alpha + " min: " + min + " max: " + max + " concept_id: " + concept_id + " normalize: " + normalize + " highlight_border: " + highlight_border);
      let color = [0, 0, 0];
      if (concept_id != null) {
        const concept = this.jsonData.concepts[concept_id];
        color = concept.color.map((c) => Math.floor(c * 255));
      }
      if (normalize) {
        alpha = (alpha - min) / (max - min);
      }
      const alpha_ratio = highlight_border ? 0.75 : 1.0;
      const border_color = [...color, alpha];
      const background_color = [...color, alpha * alpha_ratio]; // we actually dim the inside of the word to highlight the border
      const style = `background-color: rgba(${background_color.join(
        ","
      )}); outline-color: rgba(${border_color.join(",")});`;
      return style;
    }

    /**
     * Activate a concept by its id, this method is called when the mouse is over a concept.
     * Refresh the styles of the inputs and outputs according to the selected concept.
     *
     * @param {number} concept_id - The id of the concept to activate
     */
    activateConcept(concept_id) {
      console.log("Activating concept " + concept_id);
      this.current_concept_id = concept_id;
      this.refreshConceptsStyles();
      this.refreshInputsStyles();
      this.refreshOutputsStyles();
    }

    /**
     * Deactivate a concept by its id, this method is called when the mouse is not over a concept anymore.
     * If a concept was previously selected, it is reactivated.
     *
     * @param {number} concept_id - The id of the concept to deactivate
     */
    deactivateConcept(concept_id) {
      console.log("Deactivating concept " + concept_id);
      this.activateConcept(this.selected_concept_id);
    }

    /**
     * Select a concept by its id: this method is called when the user clicks on a concept.
     * If the concept is already selected, it is deselected.
     *
     * @param {number} concept_id - The id of the concept to select
     */
    selectConcept(concept_id) {
      console.log("Selecting concept " + concept_id);
      if (this.selected_concept_id === concept_id) {
        console.log("Concept already selected, delecting it");
        this.selected_concept_id = null;
        return;
      }
      this.selected_concept_id = concept_id;
      this.activateConcept(concept_id);
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
        conceptElement.concept_id = i;
        mainConceptsDiv.appendChild(conceptElement);
      }
    }

    /**
     * Refresh the styles of the concepts according to 'current_output_id' and 'topk'
     */
    refreshConceptsStyles() {
      // find the current ouput's concepts and order them by value, take the topk
      // then display them in the correct order in the concepts div
      var mainConceptsDiv = document.getElementById(this.uniqueIdConcepts);
      var conceptElements = mainConceptsDiv.children;

      // If no output is selected yet, display all the concepts
      if (this.current_output_id == null) {
        for (let i = 0; i < conceptElements.length; i++) {
          var conceptElement = conceptElements[i];
          conceptElement.style.visibility = "visible";
        }
        return;
      }

      // topk filtering when we have outputs attributions
      if (this.jsonData.outputs.attributions) {
        // Get the topk concepts for the current output
        var topk_concepts = this.getTopkConcepts(this.current_output_id);
        console.log(
          "Refreshing concepts for output " +
            this.current_output_id +
            ", using topk concepts " +
            topk_concepts
        );

        // Reoder the elements in the conceptsElements, with the topk elements in
        // first position
        conceptElements = Array.prototype.slice.call(conceptElements);
        conceptElements.sort(function (a, b) {
          var a_value = topk_concepts.indexOf(a.concept_id);
          var b_value = topk_concepts.indexOf(b.concept_id);
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
        "Refreshing the selected concept with current_concept_id:",
        this.current_concept_id
      );
      for (let i = 0; i < conceptElements.length; i++) {
        var conceptElement = conceptElements[i];
        conceptElement.classList.toggle(
          "selected-style",
          conceptElement.concept_id === this.current_concept_id
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
        "refreshInputsStyles(), current_concept_id: ",
        this.current_concept_id
      );
      // for each input word, change the style based on the value of the concept
      // TODO: use the min and max values of the concept / normalize ?
      // var min = concept.min;
      // var max = concept.max;
      var highlight_border = true;
      var mainInputsDiv = document.getElementById(this.uniqueIdInputs);
      var sentenceElements = mainInputsDiv.children;

      // iterate on sentences
      for (let i = 0; i < sentenceElements.length; i++) {
        var sentenceElement = sentenceElements[i];
        var wordElements = sentenceElement.children;

        // iterate on words & compute its alpha value based on the attributions
        for (let j = 0; j < wordElements.length; j++) {
          var wordElement = wordElements[j];
          let alpha = 0.0;
          if (
            this.current_output_id != null &&
            this.current_concept_id != null
          ) {
            alpha =
              this.jsonData.inputs[i].attributions[this.current_output_id][j][
                this.current_concept_id
              ];
          }
          // Generate the style for the word according to the alpha value
          let style = this.getStyleForWord(
            alpha,
            0,
            1,
            this.current_concept_id,
            true,
            highlight_border
          );
          wordElement.style = style;

          // Tooltip:
          // - Remove the previous tooltip if existing
          // - Add the new tooltip with the current concept value
          var previousTooltip =
            wordElement.getElementsByClassName("tooltiptext");
          if (previousTooltip.length > 0) {
            previousTooltip[0].remove();
          }
          if (this.current_concept_id != null) {
            var tooltip = document.createElement("span");
            tooltip.classList.add("tooltiptext");
            tooltip.textContent = alpha.toFixed(3);
            wordElement.appendChild(tooltip);
          }
        }
      }
    }

    /**
     * Activate an output by its id, when the mouse is over it
     * Refresh the styles of the outputs according to the selected output
     * If a concept was previously selected, it is deactivated
     *
     * @param {number} output_id - The id of the output to activate
     */
    activateOutput(output_id) {
      console.log("Activating output " + output_id);
      console.log("The selected output is: " + this.selected_output_id);
      if (this.selected_concept_id != null) {
        console.log("A concept is selected, not activating the output");
        return;
      }
      this.current_output_id = output_id;
      // When changing output, we reset the selected concept
      this.current_concept_id = null;
      this.selected_concept_id = null;
      this.refreshOutputsStyles();
      this.refreshConceptsStyles();
      this.refreshInputsStyles();
    }

    /**
     * Deactivate an output by its id, when the mouse is not over it anymore
     *
     * @param {number} output_id - The id of the output to deactivate
     */
    deactivateOutput(output_id) {
      console.log("Deactivating output " + output_id);
      if (this.selected_concept_id != null) {
        console.log("A concept is selected, not deactivating the output");
        return;
      }
      // Reactivating the current selected concept
      if (this.selected_output_id === output_id) {
        console.log("Output already selected");
        return;
      }
      console.log(
        "Reactivation of the saved selected_output_id: " +
          this.selected_output_id +
          " and concept_id: " +
          this.selected_concept_id
      );
      this.activateOutput(this.selected_output_id);
    }

    /**
     * Select an output by its id: this method is called when the
     * user clicks on an output to fix it
     *
     * @param {number} output_id - The id of the output to select
     */
    selectOutput(output_id) {
      console.log("Selecting output " + output_id);
      if (this.selected_output_id === output_id) {
        console.log("Output already selected, deselecting it");
        this.selected_output_id = null;
      } else {
        this.selected_output_id = output_id;
      }
      this.selected_concept_id = null;
      this.activateOutput(this.selected_output_id);
    }

    /**
     * Get the topk concepts for an output_id
     * @param {number} output_id - The id of the output
     * @returns An array of topk concepts
     *
     */
    getTopkConcepts(output_id) {
      console.log("Getting topk concepts for output " + output_id);
      if (this.current_output_id == null) {
        // impossible: we need an output word selected in order to compute the
        // the topk concepts for the output_id output
        return [];
      }
      var attributions =
        this.jsonData.outputs.attributions[this.current_output_id][output_id];
      console.log("Attributions for output " + output_id + ":", attributions);
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
        this.selected_output_id
      );
      for (let i = 0; i < mainOutputsDiv.children.length; i++) {
        // Update the style of each output word, based on its position relative to the current selected output
        var child = mainOutputsDiv.children[i];
        child.classList.toggle(
          "highlighted-word-style",
          i < this.current_output_id
        );
        child.classList.toggle("selected-style", i === this.current_output_id);

        // TODO: merge with the styling done in refreshInputsStyles ?
        let alpha = 0.0;
        if (this.current_concept_id != null && i < this.current_output_id) {
          alpha =
            this.jsonData.outputs.attributions[this.current_output_id][i][
              this.current_concept_id
            ];
          let style = this.getStyleForWord(
            alpha,
            0,
            1,
            this.current_concept_id,
            true,
            true
          );
          child.style = style;
        } else {
          child.style = ""; // reset the style
        }

        // Tooltip for the output word
        // - Remove the previous tooltip if existing
        // - Add the new tooltip with the current concept value
        var previousTooltip = child.getElementsByClassName("tooltiptext");
        if (previousTooltip.length > 0) {
          previousTooltip[0].remove();
        }
        if (this.current_concept_id != null && i < this.current_output_id) {
          var tooltip = document.createElement("span");
          tooltip.classList.add("tooltiptext");
          tooltip.textContent = alpha.toFixed(3);
          child.appendChild(tooltip);
        }
      }
    }
  };
})();
