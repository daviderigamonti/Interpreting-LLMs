# Papers

## [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
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

### [Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229)
- [Local Copy](PDFs/mass_editing_memory_in_a_transformer.pdf)

## [How Pre-trained Language Models Capture Factual Knowledge? A Causal-Inspired Analysis](https://aclanthology.org/2022.findings-acl.136/)
- [Local Copy](PDFs/how_pre_trained_language_models_capture_factual_knowledge_a_causal_inspired_analysis.pdf)
- [Annotated Copy](PDFs/Annotated/how_pre_trained_language_models_capture_factual_knowledge_a_causal_inspired_analysis_annotated.pdf)
- Questions
    
    - 
- Experiment 1
    - Questions
        - Which association do PLMs (Pre-Trained Language Models) depend on to capture factual knowledge?
        - How much PLMs depend on a specific group of remaining words to predict missing words in pre-training steps.
    - Conclusions
        - PLMs depend more on the positional close and highly co-occurred associations than the knowledge-dependent association to capture factual knowledge.
    - Process
        - Build a Structural Causal Model (SCM) for the missing words generation process and apply interventions on some input words to estimate their effect quantitatively.
        - Missing words are considered as outcome words, the words that hold a certain association to those as treatment words and the remaining as context words.
        - Obtain the quantitative causal effect of treatment words on outcome words by using *do()* calculus and Average Treatment Effect (ATE).
        - Align Wikipedia sentences to *(subject, predicate, object)* triplets in the KB where objects are treated as outcome words and treatment words are words that hold specific associations to those (Knowledge-Dependent [KD], Positionally Close [PC], Highly Co-Occurred [HC] or Random Association (to provide empirical support) [R]); each sentence yields 4 probing samples of the specified categories.
    - Observations
        - General trend for dependence on associations for all models of all sizes: PC > HC > KD
        - PLMs prefer the associations founded with positionally close or the highly co-occurred words to the knowledge-base clues.
- Experiment 2
    - Questions
        - Is the association on which PLMs depend effective in capturing factual knowledge?
    - Conclusions
        - Depending on the knowledge-dependent association is more effective for factual knowledge capture than positional close and highly co-occurred associations.
    - Process
        - The performance of capturing the corresponding facts in the previous experiment is probed by having PLMs fill masks on crafted queries constructed by instantiating templates on triplets.
        - The accuracy *mrr* of capturing this fact is obtained by averaging over all the predictions obtained with different queries while the consistency *con* of the capture is indicated by the percentage of the pairs of queries that have the same result and the factual knowledge capture performance *test* is evaluated by jointly examining the accuracy and the consistency.
        - Finally, the Pearson correlation coefficient between dependence and probing performance is calculated in order to reveal the effectiveness of different associations (positive correlation = better association).
    - Observations
        - Dependence on the KD association positively correlates with probing performance, dependence on the HC association has a slightly positive correlation while the PC association holds negative correlation with probing performance.
        - The more PLMs depend on the KD association, the better the PLMs can capture the corresponding factual knowledge, while relying much on the positionally close association is harmful to the probing performance.
        - The dependence measure results reveal that the PLMs depend most on the PC and least on the KD associations, while in effectiveness measure it is found that PC is the most ineffective association for factual knowledge while the KD association is the most effective.
        - The PLMs do not capture factual knowledge ideally, since they depend more on the ineffective associations than the effective one.
- Datasets
    - TREX (aligns KB triplets with Wikipedia sentences)
- Models
    - BERT
    - RoBERTa
    - SpanBERT
    - ALBERT

## [Impact of Co-occurrence on Factual Knowledge of Large Language Models](https://arxiv.org/abs/2310.08256)
- [Local Copy](PDFs/impact_of_co_occurrence_on_factual_knowledge_of_large_language_models.pdf)
- [Annotated Copy](PDFs/Annotated/impact_of_co_occurrence_on_factual_knowledge_of_large_language_models_annotated.pdf)

## [Large Language Models Struggle to Learn Long-Tail Knowledge](https://proceedings.mlr.press/v202/kandpal23a.html)
- [Local Copy](PDFs/large_language_models_struggle_to_learn_long_tail_knowledge.pdf)
- [Annotated Copy](PDFs/Annotated/large_language_models_struggle_to_learn_long_tail_knowledge_annotated.pdf)

## [Factual Probing Is [MASK]: Learning vs. Learning to Recall](https://aclanthology.org/2021.naacl-main.398/)
- [Local Copy](PDFs/factual_probing_is_mask_learning_vs_learning_to_recall.pdf)
- [Annotated Copy](PDFs/Annotated/factual_probing_is_mask_learning_vs_learning_to_recall_annotated.pdf)

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