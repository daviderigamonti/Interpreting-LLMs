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
        - Generality means that the information is retained through rephrasing, summarization and can be referenced to by the model in unrelated contexts that ask to recall it.
        - zsRE (zero-shot Relation Extraction) has been used as a baseline by other model-editing centered papers, but does not entirely fit the specific scenario of this paper: good to measure generalization but not specificity or other possibly related metrics such as fluency.
        - CounterFact (in form subject, relationship, object) measures specificity (check for semantically neighboring subjects to the fact subject that may also be affected by the fact), fluency (human evaluation).
    - Problem of asymmetric fact storing is that it may not be possible for the model to infer the inverse relationship from a fact.

### [Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229)
- [Local Copy](PDFs/mass_editing_memory_in_a_transformer.pdf)

## [How Pre-trained Language Models Capture Factual Knowledge? A Causal-Inspired Analysis](https://aclanthology.org/2022.findings-acl.136/)
- [Local Copy](PDFs/how_pre_trained_language_models_capture_factual_knowledge_a_causal_inspired_analysis.pdf)
- [Annotated Copy](PDFs/Annotated/how_pre_trained_language_models_capture_factual_knowledge_a_causal_inspired_analysis_annotated.pdf)
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
- Questions
    - Suspect that LLMs relying on co-occurrence statistics of the pre-training corpora is one of the main factors that cause incorrect store/recall of factual knowledge focusing on subject-object co-occurrence.
- Conclusions
    - Factual probing accuracy of LLMs highly correlates with subject-object co-occurrence, leading to failures in recalling rare facts; this phenomenon is independent of scale.
    - A significant portion of facts in the LAMA dataset can be recalled by simply generating the object with the highest co-occurrence count, but this process is not necessary to recall factual knowledge, is inappropriate for understanding the accurate meaning behind words and may lead to hallucinations.
- Process
    - Each fact is converted in natural language following pre-defined templates and each fact is converted into a cloze statement by masking an object.
    - Use of restricted output candidate sets to restrict LLM vocabulary following three strategies: Remove Stopwords, Gold Objects and Relation-wise Gold Objects.
    - Co-occurrences between pairs are evaluated in a minimal sufficient set, initialized as a set of subject entities in the dataset and words in the target model's vocabulary that are object candidates (words are tokenized, normalized, stopwords removed and entities with more than 3 tokens filtered out).
    - Baselines are calculated using simple term frequency statistics: Marginal Probability, Joint Probability and PMI.
    - Two rank-based metrics (hits@1 and MRR) are considered to evaluate the performance on factual knowledge probing and baselines, where models that rank ground truth objects higher are considered more knowledgeable; to measure correlation these statistics are plotted against subject-object co-occurrence counts considering two measures: the reciprocal rank of subject-object co-occurrence counts and the conditional probability of the gold object given a subject.
- Observations
    - In zero-shot settings scaling up model sizes can improve the performance on factual knowledge probing while in fine-tuned settings the effect of model sizes and restricting output candidates is marginal, implying that the models may learn appropriate candidate sets during finetuning.
    - Finetuning improves factual knowledge probing accuracy substantially and most of the models, except for the smallest one are capable of memorizing most of the seen facts. This implies that memorization is necessary to recall facts since factual knowledge in the test set may not be inferred based on prior knowledge of other facts in the training set.
    - A large portion of the facts can be recalled with the joint probability baseline when the output candidates are tightly restricted in the gold objects (relation-wise) setting. Co-occurrence statistics may inflate model performance when evaluating LLMs even though it may not be appropriate for understanding the words' semantics.
    - In both zero-shot and finetuned settings, hits@1 is lower as the co-occurrence count is lower. Consequently, LLMs suffer from generalizing to recalling rare facts. Finetuning does not resolve co-occurrence bias despite improving the overall performance and model size is irrelevant.
    - LLMs struggle to learn facts that rarely appear in the pre-training corpora, although they are explicitly given.
    - Co-occurrence statistics are necessary but not sufficient to recall facts, therefore, a heavy reliance on co-occurrence may be problematic. In addition, a word with higher co-occurrence counts overrides the correct answer in a total of 38% of the failure cases making recalling rare facts especially difficult.
- Datasets
    - LAMA-TREx (fact represented as subject-relation-object triples, contains 41 distinct relations)
    - Penn Treebank (tokenization)
    - The Pile (pretraining of open source GPT-3 models)
- Models
    - Open Source GPT-3 models: GPT-Neo 125M, GPT-Neo 1.3B, GPT-Neo 2.7B, and GPT-J 6B
    - GPT-3.5 (InstructGPT) 175B, ChatGPT-3.5-turbo, and ChatGPT-4 (further correlation analysis testing)

## [Large Language Models Struggle to Learn Long-Tail Knowledge](https://proceedings.mlr.press/v202/kandpal23a.html)
- [Local Copy](PDFs/large_language_models_struggle_to_learn_long_tail_knowledge.pdf)
- [Annotated Copy](PDFs/Annotated/large_language_models_struggle_to_learn_long_tail_knowledge_annotated.pdf)
- Experiment 1
    - Questions
        - What is the relationship between the knowledge learned by an LLM and the information in its pretraining dataset?
    - Conclusions
        - There is a strong correlation between an LM’s ability to answer a question and the number of pre-training documents relevant to that question for numerous QA datasets, pretraining datasets, and model size.
        - Model accuracy drops significantly on questions whose relevant documents were removed, which shows that the observed correlational trends are likely causal in nature.
    - Process
        - Evaluation of models on the curated, linked, question-answering dataset using 4-shot evaluation and by greedily decoding answers until the models generate a newline character.
        - Relevant documents are identified by searching for co-occurrences of salient question and answer entities, this method is compared against two baselines: counting documents that contain the salient question entity and counting documents that contain the salient answer entity.
        - To address the hypothesis of rare question being harder, human evaluation is also taken into consideration on the task.
        - Two versions of the same 4.8B parameters model are trained, a baseline on C4 and a “counterfactual” LM on a modified pre-training dataset. This is to measure the effect of deleting certain documents from the training set by comparing the performance of the two models.
    - Observations
        - There is a strong correlation between question answering accuracy and relevant document count for all tested models. Correspondingly, when the number of relevant documents is low, models are quite inaccurate. Model size is also strongly correlated with question answering performance.
        - All considered document identification methods are correlated with QA accuracy. However, when only considering QA examples where the question and answer entities co-occur few (<5) times the two baseline methods no longer correlate with QA accuracy, this indicates that counting documents with just the answer entity or question entity alone is insufficient for explaining why LMs are able to answer certain questions.
        - Human accuracy is highest for the questions with few relevant documents, the opposite trend of models. Humans may be better on questions with few relevant documents because questions about rarer facts are more likely to be simple factoids compared to common entities, and the Wikipedia documents are that are provided to the annotators are shorter for rarer entities, which makes reading comprehension easier.
        - When comparing the effects of a model trained on a dataset with missing documents, for questions with few relevant documents in the original C4 dataset, performance is poor for both the baseline and the counterfactual LM. However, for questions with many relevant documents, performance is significantly worse for the counterfactual LM. This suggests a causal link between the number of relevant documents and QA performance.
- Experiment 2
    - Questions
        - Are there ways to better capture knowledge that rarely appears in the pre-training data?
    - Conclusions
        - For model scaling, there is a strong log-linear relationship between parameter count and QA accuracy. These trends show that while scaling up LMs improves knowledge learning, models would need to be scaled dramatically to achieve competitive QA accuracy on long-tail questions.
        - Retrieval-augmented systems are more promising when a retriever succeeds in finding a relevant document, it reduces an LLM’s need to have a large amount of relevant pretraining text. Nevertheless, retrieval systems themselves still exhibit a mild dependence on relevant document count.
    - Process
        - First naive approaches for improving accuracy on questions about less-prevalent knowledge is to collect larger quantities of data, increase the diversity of the pre-training data or use larger models.
        - An alternative option is to directly modify the training objective to encourage memorization, this can be accomplished by increasing the number of training epochs, directly modifying the loss to encourage the model to focus on salient facts or designing a curriculum to minimize forgetting.
        - For knowledge-intensive tasks, a natural alternative is to make LMs retrieval augmented by combining them with a retrieval module that returns relevant textual contexts, in this case both Oracle Retrieval (300-word segment that surrounds the ground-truth answer from the gold Wikipedia page) and BM25 Retrieval (BM25 retriever to select relevant paragraphs from Wikipedia) are analyzed.
    - Observations
        - Increasing data quantity would not significantly improve accuracy as scaling datasets by moderate factor usually results in small accuracy gains and increasing diversity would also provide minimal benefit because many data sources are surprisingly correlated.
        - Using larger models consistently produces better QA performance. However, one would need immensely large LMs to achieve high accuracy on long-tail questions.
        - Oracle retrieval-augmentation dramatically boosts accuracy over closed-book models, especially on rarer instances. QA accuracy actually goes down as the number of relevant documents increases—the opposite trend of closed-book LLMs, humans exhibit the same trend.
        - BM25 attains reasonably high recall, especially for larger values of k. However, the BM25 retriever still shows a mild dependence on relevant document count, while the BM25 retrieval-augmented models outperform their closed-book counterparts across all ranges of relevant document counts, and especially on rare examples.
- Datasets
    - DBpedia Spotlight Entity Linker (entity linking on datasets) + human evaluation of the pipeline
    - The Pile, ROOTS (En), C4, OpenWebText, Wikipedia (massive pre-training corpora)
    - Natural Questions, TriviaQA (QA datasets)
- Models
    - GPT-Neo, GPT-NeoX, and GPT-J LM (trained on The Pile)
    - BLOOM (trained on ROOTS)
    - GPT-3 Ada, GPT-3 DaVinci

## [Factual Probing Is [MASK]: Learning vs. Learning to Recall](https://aclanthology.org/2021.naacl-main.398/)
- [Local Copy](PDFs/factual_probing_is_mask_learning_vs_learning_to_recall.pdf)
- [Annotated Copy](PDFs/Annotated/factual_probing_is_mask_learning_vs_learning_to_recall_annotated.pdf)
- Experiment 1
    - Questions
        - Research for better prompts.
        - Restricting the search to the space of vocabulary tokens is a suboptimal and artificial constraint.
    - Conclusions
        - Rather than confining the search space to discrete input tokens, it is possible to directly optimize in the input embedding space, finding the real-valued input vectors that are most effective at eliciting facts and that initializing with manual prompts can provide a better starting point for the search process -> OPTIPROMPT approach.
    - Process
        - Idea of AUTOPROMPT but on the continuous embeddings space, rather than the discrete token space.
        - A prompt is defined as a series of dense vectors in the input embedding space. Gradient-descent is used to minimize the negative log-likelihood of the prompt over a training set; the length of the prompt is treated as a hyperparameter and the vectors are either randomly or manually initialized.
        - OPTIPROMPT is trained on the same data as AUTOPROMPT.
    - Observations
        - OPTIPROMPT outperforms AUTOPROMPT on the LAMA and LAMA-UHN benchmarks with a consistent improvement on every category, except for the “1-1” category, the prompt that yields the best results in this category is the manual prompt, with LPAQA and AUTOPROMPT prompts performing steadily worse. There may be very few prompts that elicit this relation with high accuracy, and they are difficult to find via stochastic, non-convex optimization.
        - Initializing the prompt vectors using the manually written prompts improves performance consistently, thus manual initialization is likely to provide a good prior for finding a good solution in the non-convex optimization problem. 
- Experiment 2
    - Questions
        - Prompts that are optimized on training data may exploit some regularities in the underlying distribution of facts. How can we make sure our prompts are recovering information solely from the language model?
    - Conclusions
        - All the data-driven prompt-search methods, including previous methods and the proposed OPTIPROMPT are able to exploit regularities to achieve better prediction accuracy.
        - Given some training data, a good search algorithm can find prompts that recover a non-trivial number of “facts” from a neural network with randomly initialized parameters exploiting both simple class statistics and higher order lexical regularities.
    - Process
        - Candidate patterns to be found are the class prior $P(o|r)$ (if one or two object labels dominate the relation $r$, it is easier to guess them regardless of the subject entity) and the correlation between subject tokens and object labels, that is, to estimate $P(o|r,w_1,...,w_{|s|})$ where $w_1,...,w_{|s|} \in \mathcal{V}$ are the tokens of the subject name.
        - Fit two simple probabilistic models to check the existence of the previous patterns in the Wikidata training set: the first model always predicts the majority class, with class priors learned from the training data, and the second is a Naive Bayes classifier with add-one smoothing.
        - To establish if a prompt optimization method built with pre-trained language models is expressive enough to exploit the previously identified regularities in practice, two random control trials in the form of model baselines are used.
        - The Random Model (RM) baseline optimizes prompts to elicit facts from a neural network with the same architecture as the pre-trained LM but with randomly initialized parameters, thus any successful predictions in this setting must be the result of optimizing on training data. While the Random Embeddings (RE) baseline is analogous to the previous one, but only the input embeddings are reinitialized.
        -  Finally, a reinitialized BERT model is fine-tuned on the training data with the goal of getting a better estimate of the number of LAMA facts that could be predicted from the training data.
    - Observations
        - The majority class probabilistic model performs well because, on some relations, well over half of the examples are from the majority class. The Naive Bayes probabilistic baseline performs even better in all categories by learning correlations between subject tokens and object labels.
        - In the RE setting, both AUTOPROMPT and OPTIPROMPT are capable of finding prompts that elicit some correct predictions, while in the RM setting, AUTOPROMPT gets no prediction correct, presumably because it is harder to optimize, but OPTIPROMPT is still capable of finding successful prompts.
        - Most successful predictions are obtained by finding a prompt that elicits the majority class label.
        - Fine-tuning BERT results in even higher accuracy, indicating that there are patterns that prompts fail to exploit.
        - The random controls represent a challenging setting for prompt optimization, and it is possible that the prompts are better exploiting the training data when they have access to full pretrained BERT model. We find evidence that this is the case by calculating how often each prompt elicits the training class majority label on LAMA. Both AUTOPROMPT and OPTIPROMPT are prone to over-predicting the majority class label.
- Experiment 3
    - Questions
        - How can factual probing results be interpreted?
    - Conclusions
        - Specific control experiments can be used to form a more detailed understanding of the behavior of different probes and forming some conclusions about which facts are less likely to have been learned from training data.
    - Process
        - In order to get another perspective of the relative improvement, the LAMA dataset is partitioned into an easy subset (facts that can be correctly predicted by any of three models fit to the training data) and a hard subset (the remaining facts). The easy subset serves as an estimate of the set of facts that can be predicted from training data.
    - Observations
        - All probing methods achieve a much higher accuracy on the easy subset.
        - Using more sophisticated prompt optimization techniques tends to result in big improvements on the easy subset of LAMA and smaller improvements on the hard subset.
        - OPTIPROMPT outperforms AUTOPROMPT on the easy examples; while still yielding a big improvement on the hard examples, this suggests that OPTIPROMPT is both better at learning from training data and better at eliciting facts from an LM.
        - In the easy subset, both AUTOPROMPT and OPTIPROMPT elicit more accurate predictions on cases when the answer is a token in the subject name, while in the hard subset they show signs of having over-fit to the training distribution, incorrectly predicting the most common object labels. They both appear to be exploiting training data.
        - A more general limitation of this analysis is that it does not allow to say which strategy a model uses to make a particular prediction. Many facts can be predicted either by learning the class prior, by learning a lexical correlation between subject tokens and objects, by exploiting lexical information from the LM, or because the LM genuinely encodes information about a particular entity.
- Datasets
    - LAMA benchmark (<s,r,o> triplets from Wikidata, ConceptNet and SQuAD)
    - TREX (split of LAMA benchmark with relationships divided into 1-1, N-1, 1-N and N-N)
    - LPAQA (training dataset from Wikidata with no overlap with <s,o> pairs in LAMA)
    - AUTOPROMPT (statistical model that searches for <s,r,o> triples and creates prompts)
- Models
    - LAMA (model for the LAMA benchmark)

# Explainability Libraries

## Captum
- [-] Not really optimized for autoregressive models.
- IntegratedGradients is an axiomatic model interpretability algorithm that assigns an importance score to each input feature by approximating the integral of gradients of the model's output w.r.t. the inputs along the path from given baselines/references to input.
- LayerIntegratedGradients isn't easy to interpret
- providing different baselines yields very different results.

## Ecco
- [-] Doesn't work on Kaggle and Google Colab notebooks.

## TruLens
- [-] Too high level, unrelated to our task.

## Alibi Explain
- [+] General explainability framework with *Anchors* and *Integrated Gradients* modules that are focused on text.
- [-] Not really optimized for autoregressive models.

## SHAP
- [+] Can be used seamlessly with autoregressive models.
- [-] Explanations are provided through shapley values, does not look at gradient updates.

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