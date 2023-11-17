# Papers

### [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
- [From Week 3](../Week3/Resources.md)
- [Local Copy](../Week3/PDFs/locating_and_editing_factual_associations_in_gpt.pdf)
- [Annotated Copy](../Week3/PDFs/Annotated/locating_and_editing_factual_associations_in_gpt_annotated.pdf)
- [Website](https://rome.baulab.info)
- [Interview](https://www.youtube.com/watch?v=_NMQyOu2HTo)

# Notebooks Ideas

## String ID Detokenization
- Idea: Try `tokenizer.deconde(id_string)` to shed light onto BPE for Phi-1.5.
- Results: Discovered that in phi-1.5 the byte-pair encoding, encoded words like cat and dog including a leading whitespace character, therefore the token for "dog" and " dog" are different. This, however doesn't affect the previous findings.

## Capitalization Variants for Swapping Experiment
- Idea: Try different combinations for uppercase/lowercase swapping combinations for animal queries.
    - Leading whitespace lowercase
    - Leading whitespace lowercase + No leading whitespace lowercase
    - No leading whitespace lowercase
    - Whitespace uppercase
- Results: Tests with only uppercase/lowercase embedding swap seem to point to the fact that embeddings that are not present in the input do not influence the model prediction at this scale.

## Add Targeted Noise to Embeddings
- Idea: Add noise to subjects of the animal queries once at a time.
- Results: Model associates multiple words to the perturbed cat/dog embedding depending on the question made. When swapping and adding noise simultaneously results may not be consistent between the swapped terms.

## Look at Saliency Over the Input
- Idea: Look at the gradient produced by words over all the input (multiply gradient for the embedding size?).
- Libraries:
    - [Captum](https://captum.ai)
    - [Ecco](https://github.com/jalammar/ecco)
- Results: 

# Previous Meeting Ideas

- Study benchmark usage in literature
- Experiment 1
    - Fine-tune model on new, never-before-seen facts
    - Observe where gradient has more effect on the model weights
    - Example: 
        - Propose sentence with new fact to the model
        - Make one gradient update and observe it
        - Ask question regarding new fact and see if answer contains the expected results
        - Repeat until model answers properly
- Experiment 2
    - Freeze/Unfreeze model layers
    - Repeat fine-tuning
- Important to evaluate model on benchmarks after experiments, to check if updates have been destructive towards the model baseline on common tasks.