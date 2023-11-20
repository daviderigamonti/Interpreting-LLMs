# Papers

### [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
- [From Week 3](../Week3/Resources.md)
- [Local Copy](../Week3/PDFs/locating_and_editing_factual_associations_in_gpt.pdf)
- [Annotated Copy](../Week3/PDFs/Annotated/locating_and_editing_factual_associations_in_gpt_annotated.pdf)
- [Website](https://rome.baulab.info)
- [Interview](https://www.youtube.com/watch?v=_NMQyOu2HTo)
    - Overarching research topic: Mechanistic Interpretability Research
    - Causal Tracing Approach
        - When throughout an input sentence and the model architecture, does the model realize the answer to the sentence?
        - Replacing a hidden state of a corrupted run with the same one from the clean run should demonstrate that the specific hidden state is associated with the fact if it is able to restore the correct answer.
        - Patch of high causal effect in MLPs right after the last token of the subject is read.
        - Hypothesis: the MLPs are storing factual associations that are not portrayed in the embedding -> linear associative memory.
        - Idea:[Try repeating their experiment but remove association between object and answer in embedding and see if the MLPs are still able to recover the factual association]
        - Idea:[Compare the ease of MLPs in retrieving information for (subject, answer) pairs that have related embeddings and unrelated embeddings]
        - MLP as key-value store memory due to the nice properties of having two matrices (fan-out and fan-in) joined by nonlinear functions.
    - ROME
        - Key-value store associates a subject (key) with an information (value) which is not necessarily encoded with the output embedding and is not necessarily the needed information.
        - Insertion of key-value pairs into the MLP weights:
            - The desired key is obtained by observing the MLP inputs when the specific subject is mentioned (possibly averaging the results through multiple contexts).
            - The desired value is obtained solving a constrained optimization problem consisting of trying to make the MLP output the desired relationship.
            - "Essence drift" problem of changing a subject's fundamental properties when editing a relationship, a specific term in the value optimization problem is dedicated to reduce this effect.
        - Constrained minimization approach is chosen so that all old key and values that are not of interest "cancel out" and it isn't needed to discover them.
        - How to choose the MLP to perform ROME on? Infer a range of interested MLPs with Causal Tracing (since it possible that multiple MLP store the same fact redundantly and all collaborate towards retrieving it) and choose the MLP where there is a peak in the Causal Tracing analysis.
        - Possible scaling issues when inserting multiple facts.
        - Follow-up paper on [Mass-Editing Memory in a Transformer](#mass-editing-memory-in-a-transformer).
    - ROME Evaluation
        - Specificity means that when inserting a new single piece information into the model shouldn't "learn" new information that wasn't specified; e.g.: intended information: the cat barks -> possible unintended information: the monkey also barks.
        - Generality means that the information is retrained through rephrasing, summarization and can be referenced to by the model in unrelated contexts that ask to recall it.
        - zsRE (zero-shot Relation Extraction) has been used as a baseline by other model-editing centered papers, but does not entirely fit the specific scenario of this paper: good to measure generalization but not specificity or other possibly related metrics such as fluency.
        - CounterFact (in form subject, relationship, object) measures specificity (check for semantically neighboring subjects to the fact subject that may also be affected by the fact), fluency (human evaluation).
    - Problem of asymmetric fact storing is that it may not be possible for the model to infer the inverse relationship from a fact.

## [Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229)
- [Local Copy](PDFs/mass_editing_memory_in_a_transformer.pdf)

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

# Previous Meetings Ideas

## 14/11/23 w/ Mark, Nicolò, Vincenzo

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

## 20/11/23 w/ Nicolò

- Hallucination survey paper containing multiple interesting papers in the factuality section, explore state-of-the-art.
    - Study papers focusing on Goal and Experiments (Input + Conditions + Outputs/Findings).

- Search additional explainability libraries focused on gradient tracing on autoregressive models/LLMs.

- Ecco
    - Explore possibilities and API usage, if the investment is worthwhile, try to run it.
- Captum
    - Try saliency evaluation different query baselines (SOS, EOS, ...).
    - Try saliency evaluation combined with embedding swap