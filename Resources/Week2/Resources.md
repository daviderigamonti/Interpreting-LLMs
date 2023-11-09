# Related Work

## Noise In Hidden Representation For Reducing Overfitting

### [Fine-tuning Pre-trained Language Models with Noise Stability Regularization](https://arxiv.org/pdf/2206.05658.pdf)
- [Local Copy](PDFs/fine_tuning_pre_trained_language_models_with_noise_stability_regularization.pdf)
- Add noise to regularize hidden representations of network: Layerwise Noise Stability Regularization
- Applied to reduce overfitting of a LLM during fine-tuning
- Layer-wise regularization that explicitly enforces noise stability of middle layers and In-manifold noise stability regularization.

### [HyPe: Better Pre-trained Language Model Fine-tuning with Hidden Representation Perturbation](https://arxiv.org/pdf/2212.08853.pdf)
- [Local Copy](PDFs/Interest/hype_better_pre_trained_lanuage_model_fine_tuning_with_hidden_representation_perturbation.pdf)
- [Annotated Copy](PDFs/Interest/Annotated/hype_better_pre_trained_lanuage_model_fine_tuning_with_hidden_representation_perturbation_annotated.pdf)
- Add noise to hidden representations of a network
- Applied to reduce overfitting of a LLM during fine-tuning
- Performs well on small datasets but on larger datasets the performance gain becomes negligible
- Overlaps with dropout regularization, therefore performance drops when used in combination
- Uniform and Gaussian noise

## Noise In Input Layers For Reducing Overfitting

### [NEFTune: Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/pdf/2310.05914.pdf)
- [Local Copy](PDFs/neftune_noisy_embeddings_improve_instruction_finetuning.pdf)
- Add noise to input word embeddings of a network
- Focus on conversational tasks
- Applied to reduce overfitting of a LLM during fine-tuning

### [Learning to Perturb Word Embeddings for Out-of-distribution QA](https://arxiv.org/pdf/2105.02692.pdf)
- [Local Copy](PDFs/Interest/learning_to_perturb_word_embeddings_for_out_of_distribution_qa.pdf)
- [Annotated Copy](PDFs/Interest/Annotated/learning_to_perturb_word_embeddings_for_out_of_distribution_qa_annotated.pdf)
- Learn perturbations by using a feedback process
- Apply learned perturbations to input embeddings in order to change their value without changing the semantics
- Applied to perform data augmentation and reduce overfitting for a QA and QG tasks
- Demonstration of how this can be considered a particular form of dropout regularization
- Rich of interesting visualizations of the perturbations on input data
- Added qualitative/quantitative study of effect of perturbations on words when projected back from embedding
- [This](https://arxiv.org/pdf/1804.08166.pdf) is a similar paper, less in depth 
    - [Local Copy](PDFs/word_embedding_perturbation_for_sentence_classification.pdf)
    - Present several perturbation methods on word embedding layer, such as Gaussian noise, Bernoulli noise and adversarial training.

### [Incorporating Noisy Length Constraints into Transformer with Length-aware Positional Encodings](https://aclanthology.org/2020.coling-main.319.pdf)
- [Local Copy](PDFs/Interest/incorporating_noisy_length_constraints_into_transformer_with_length_aware_positional_encodings.pdf)
- [Annotated Copy](PDFs/Interest/Annotated/incorporating_noisy_length_constraints_into_transformer_with_length_aware_positional_encodings_annotated.pdf)
- Noise added to length constraints of LRPE and LDPE positional encoding
- Noise as a uniform random integer inside a fixed window of integers 
- Focus on reducing the problem of under-translation in the Neural Machine Translation task

## Input Perturbation For Improving Explainability

### [Perturbing Inputs for Fragile Interpretations in Deep Natural Language Processing](https://arxiv.org/pdf/2108.04990.pdf)
- [Local Copy](PDFs/perturbing_inputs_for_fragile_interpretations_in_deep_natural_language_processing.pdf)
- Explainability-oriented
- Focus on manipulating interpretations via manual perturbations
- Input sentence is perturbed directly, before embedding

### [Sharp Nearby, Fuzzy Far Away: How Neural Language Models Use Context](https://arxiv.org/pdf/1805.04623.pdf)
- [Local Copy](PDFs/Interest/sharp_nearby_fuzzy_far_away_how_neural_language_models_use_context.pdf)
- [Annotated Copy](PDFs/Interest/Annotated/sharp_nearby_fuzzy_far_away_how_neural_language_models_use_context_annotated.pdf)
- Focuses only on pure LSTM architectures
- Good insight into performing ablation analysis on language models
- Measure increase in perplexity when perturbations are made
- Mainly focuses on context size, length and meaning of position
- Input sentence is perturbed directly, before embedding

##  Generalized Perturbations for Improving Explainability

### [NLIZE: A Perturbation-Driven Visual Interrogation Tool for Analyzing and Interpreting Natural Language Inference Models](https://www.osti.gov/pages/servlets/purl/1562803)
- [Local Copy](PDFs/Interest/nlize_a_perturbation_driven_visual_interrogation_tool_for_analyzing_and_interpreting_natural_language_inference_models.pdf)
- [Annotated Copy](PDFs/Interest/Annotated/nlize_a_perturbation_driven_visual_interrogation_tool_for_analyzing_and_interpreting_natural_language_inference_models_annotated.pdf)
- Explainability-oriented
- Perform dynamical adjustments to the model and observe the results to discover how it works
- Focused on the Natural Language Inference task
- Various kinds of perturbations
- Propose visualization framework (NLIZE) and optimization of MIRA-based error-correction
- Implementation-oriented, proposes a system by explaining its features and use cases

## Removing Layers For Overfitting

### [On the Effect of Dropping Layers of Pre-trained Transformer Models](https://arxiv.org/pdf/2004.03844.pdf)
- [Local Copy](PDFs/Interest/on_the_effect_of_dropping_layers_of_pre_trained_transformer_models.pdf)
- [Annotated Copy](PDFs/Interest/Annotated/on_the_effect_of_dropping_layers_of_pre_trained_transformer_models_annotated.pdf)
- Use activation patterns and weights to find layers that contribute less to a prediction
- Different criteria for dropping layers compared
- Efficient use of transformer parameters by removing layers
- Analyze how different models/tasks perform with fewer layers
- Focus on retraining performance

### [Reducing Transformer Depth on Demand with Structured Dropout](https://arxiv.org/pdf/1909.11556.pdf)
- [Local Copy](PDFs/reducing_transformer_depth_on_demand_with_structured_dropout.pdf)
- Randomly dropping layers at training time -> Layerdrop
- Requires additional training 
- Focus on reducing overfitting

## Model Interpretation

### [Neuron-level Interpretation of Deep NLP Models: A Survey](https://arxiv.org/abs/2108.13138)
- [Local Copy](PDFs/Interest/neuron_level_interpretation_of_deep_nlp_models_a_survey.pdf)
- [Annotated Copy](PDFs/Interest/Annotated/neuron_level_interpretation_of_deep_nlp_models_a_survey_annotated.pdf)
- Summary of major methods and findings related to studying deep NLP models at neuron-level
- Task of explaining which function each neuron carries out inside a model and which layers are relevant to certain tasks
- In depth overview of most approaches explained in various papers 
- Considerations over possible future work in this field

### [What Happens To BERT Embeddings During Fine-tuning?](https://arxiv.org/abs/2004.14448)
- [Local Copy](PDFs/Interest/what_happens_to_bert_embeddings_during_fine_tuning.pdf)
- [Annotated Copy](PDFs/Interest/Annotated/what_happens_to_bert_embeddings_during_fine_tuning_annotated.pdf)
- Explore pre-trained BERT behavior during fine-tuning from a layer-wise perspective
- Analyze if encoding of linguistic features (syntactic/semantic roles) is retained from base model
- Find which layers are more affected by fine-tuning
- Observe if these changes are generalizable or domain-specific (w.r.t. fine-tuning domain)
- Probing, ablation and RSA are the techniques of choice to tackle the problem 

## Other

### [ROBUSTLR: A Diagnostic Benchmark for Evaluating Logical Robustness of Deductive Reasoners](https://arxiv.org/pdf/2205.12598.pdf)
- [Local Copy](PDFs/robustlr_a_diagnostic_benchmark_for_evaluating_logical_robustness_of_deductive_reasoners.pdf)
- Effect of logical perturbations on deductive reasoning benchmarks
- More oriented towards "making a model learn logic reasoning"

### [How to manually add noise to embeddings for RoBERTa?](https://discuss.huggingface.co/t/how-to-manually-add-noise-to-embeddings-for-roberta/50150/2)
- Forum post on HuggingFace forums
- User wants to add noise to RoBERTa's embeddings


# Benchmarks

### General Language Understanding Evaluation (GLUE):
- https://paperswithcode.com/dataset/glue
- Tasks: Text Classification, Sentiment Analysis, Semantic Textual Similarity, Natural Language Inference, Linguistic Acceptability, Paraphrase Identification
- Includes:
    - CoLA, SST-2 (Single-Sentence Tasks)
    - MRPC, STS-B, QQP (Similarity and Paraphrasing Tasks)
    - MNLI, QNLI, RTE, WNLI (Natural Language Inference)

### Super General Language Understanding Evaluation (SuperGLUE):
- https://paperswithcode.com/dataset/superglue
- Tasks: Natural Language Inference, Question Answering, Common Sense Reasoning, Coreference Resolution, Word Sense Disambiguation
- Includes:
    - RTE, CommitmentBank (Natural Language Inference Task)
    - BoolQ, COPA, MultiRC (Question Answering Task)
    - ReCoRD (Common Sense Reasoning Task)
    - WSC (Coreference Resolution Task)
    - WiC (Word Sense Disambiguation)

### Adversarial Natural Langauge Inference (ANLI)
- https://paperswithcode.com/dataset/anli
- Task: Natural Language Inference

### Stanford Natural Language Inference (SNLI)
- https://paperswithcode.com/dataset/snli
- Tasks: Natural Language Inference

### The Stanford Question Answering Dataset (SQuAD)
- https://paperswithcode.com/dataset/squad
- Tasks: Question Answering

### CoNLL-2003 NER dataset
- https://paperswithcode.com/dataset/conll-2003
- Tasks: Token Classification, Named Entity Recognition

### CNN/Daily Mail
- https://paperswithcode.com/dataset/cnn-daily-mail-1
- Task: Abstractive Text Summarization, Seq2Seq Language Modeling, Question Answering, Text Summarization, Document Summarization, Extractive Summarization

### Extreme Summarization (XSum)
- https://paperswithcode.com/dataset/xsum
- Tasks: Seq2Seq Language Modeling, Summarization, Text Summarization

### BigPatent
- https://paperswithcode.com/dataset/bigpatent
- Task: Summarization, Text Summarization

### Microsoft Machine Reading Comprehension Dataset (MS MARCO)
- https://paperswithcode.com/dataset/ms-marco
- Passage Retrieval, Passage Ranking, Passage Re-Ranking, Question Answering

### WebQuestions
- https://paperswithcode.com/dataset/webquestions
- Tasks: Question Answering, KG-to-Text Generation, Open-Domain Question Answering

### IMDb Movie Reviews
- https://paperswithcode.com/dataset/imdb-movie-reviews
- Tasks: Text Classification, Sentiment Analysis

### English Penn Treebank (PTB)
- https://paperswithcode.com/dataset/penn-treebank
- Tasks: Language Modelling, Constituency Parsing, Dependency Parsing, Part-Of-Speech Tagging, Constituency Grammar Induction, Chunking, Unsupervised Dependency Parsing

### Natural Questions 
- https://paperswithcode.com/dataset/natural-questions
- Tasks: Question Answering, Passage Retrieval, Open-Domain Question Answering

### WikiText
- https://paperswithcode.com/dataset/wikitext-2
- Tasks: Language Modelling

### Microsoft Research Paraphrase Corpus (MRPC)
- https://paperswithcode.com/dataset/mrpc
- Tasks: Semantic Textual Similarity

### DBpedia
- https://paperswithcode.com/dataset/dbpedia
- Tasks: Text Classification

### TriviaQA
- https://paperswithcode.com/dataset/triviaqa
- Tasks: Question Answering, Open-Domain Question Answering

### HotpotQA
- https://paperswithcode.com/dataset/hotpotqa
- Tasks: Question Answering 

### ReAding Comprehension dataset from Examinations (RACE)
- https://paperswithcode.com/dataset/race
- Tasks: Question Answering, Reading Comprehension

### Multi-Perspective Question Answering Opinion Corpus (MPQA)
- https://paperswithcode.com/dataset/mpqa-opinion-corpus
- Tasks: Sentiment Analysis, Fine-Grained Opinion Analysis

### CommonsenseQA 
- https://paperswithcode.com/dataset/commonsenseqa
- Tasks: Common Sense Reasoning

### Cross-lingual TRansfer Evaluation of Multilingual Encoders (XTREME)
- https://paperswithcode.com/dataset/xtreme
- Tasks: Token Classification, Zero-Shot Cross-Lingual Transfer 

### EAI LM evaluation harness
- https://github.com/EleutherAI/lm-evaluation-harness

### HumanEval
- https://paperswithcode.com/dataset/humaneval
- Tasks: Code Generation, Code Synthesis

### Massive Multitask Language Understanding (MMLU)
- https://paperswithcode.com/dataset/mmlu
- Tasks: Multitask Language Understanding

## Notable Mentions

### Code

- Mostly Basic Python Programming (MBPP)
    - https://paperswithcode.com/dataset/mbpp
    - Tasks: Code Generation

### Question Answering

- OpenBookQA
    - https://paperswithcode.com/dataset/openbookqa
    - Tasks: Question Answering

- Social Interaction QA (SIQA)
    - https://paperswithcode.com/dataset/social-iqa
    - Tasks: Question Answering

- Physical Interaction QA (PIQA)
    - https://paperswithcode.com/dataset/piqa
    - Tasks: Question Answering

- Question Answering in Context (QuAC)
    - https://paperswithcode.com/dataset/quac
    - Tasks: Question Answering

### Common Sense Reasoning

- AI2 Reasoning Challenge (ARC)
    - https://paperswithcode.com/dataset/arc
    - Tasks: Common Sense Reasoning

- HellaSwag 
    - https://paperswithcode.com/dataset/hellaswag
    - Tasks: Sentence Completion

### Math

- GSM8K
    - https://paperswithcode.com/dataset/gsm8k
    - Tasks: Arithmetic Reasoning

- MATH
    - https://paperswithcode.com/dataset/math
    - Tasks: Math Word Problem Solving 


# Models

### [The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only](https://arxiv.org/pdf/2306.01116.pdf)
- [Local Copy](PDFs/Interest/the_refinedweb_dataset_for_falcon_llm_outperforming_curated_corpora_with_web_data_and_web_data_only.pdf.pdf)
- [Annotated Copy](PDFS/Interest/Annotated/the_refinedweb_dataset_for_falcon_llm_outperforming_curated_corpora_with_web_data_and_web_data_only_annotated.pdf)
- Main paper topic is the creation of the RefinedWeb dataset for model training, using the MDR (MacroData Refinement) pipeline that makes use of the following design principles:
    - Scale first
    - Strict dedupilcation
    - Neutral filtering
- Author uses Falcon LLM model trained on the RefinedWeb dataset and compares it with other state-of-the art models
- Model cards
    - Falcon-RW
        - Type: Autoregressive Transformer model trained with a causal language modeling objective
        - Architecture: 
            - Configuration and hyperparameters based on [GPT-3](https://arxiv.org/abs/2005.14165)
            - [ALiBi positional encodings](https://arxiv.org/abs/2108.12409)
            - [FlashAttention](https://arxiv.org/abs/2205.14135)
        - Parameters: 1B / 3B / 7B
        - Training datasets
            - RefinedWeb (350B tokens out of 5T tokens (600B version publicly available))
        - Languages: English

### [Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality](https://lmsys.org/blog/2023-03-30-vicuna/)
- Model cards
    - Vicuna-13B
        - Type: Autoregressive Transformer model trained with a causal language modeling objective
        - Architecture
            - Configuration and hyperparameters based on [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)
            - Context Length: 2k
            - Training loss adjusted for multi-turn conversations
            - [Gradient Checkpointing](https://arxiv.org/abs/1604.06174)
            - [FlashAttention](https://arxiv.org/abs/2205.14135)
        - Parameters: 13B
        - Training datasets
            - User-shared ShareGPT conversations (70K samples)
        - Languages: Primarily English
        - Notes
            - Fine-tuned from LLaMA following Stanford Alpaca
            - Scalable infrastructure
        
### [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [Local Copy](PDFs/Interest/llama_2_open_foundation_and_fine_tuned_chat_models.pdf)
- Model cards
    - LLaMA2
        - Type: Autoregressive Transformer model trained with a causal language modeling objective
        - Architecture
            - Context Length: 4k
            - LR: $3.0 \times 10^{-4}$ (7B and 13B) / $1.5 \times 10^{-4}$ (70B)
            - [Pre-normalization using RMSNorm](https://arxiv.org/abs/1910.07467)
            - [SwiGLU activation functions](https://arxiv.org/abs/2002.05202)
            - [Rotary positional embeddings](https://arxiv.org/abs/2104.09864)
            - [AdamW optimizer](https://arxiv.org/abs/1711.05101) with $\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-5}$
            - Cosine learning rate schedule with 2k steps warmup and 1.0 decay
            - [BytePair encoding tokenization (SentencePiece)](https://arxiv.org/abs/1808.06226) with total vocabulary size equal to 32k tokens
            - [Grouped-query attention](https://arxiv.org/abs/2305.13245)
        - Parameters: 7B / 13B / 70B
        - Training datasets
            - Publicly available datasets (2T tokens)
        - Languages: Primarily English
        - Notes
            - Updated version of LLaMA 1
            - Numbers are split into individual digits and unknow UTF-8 characters are decomposed into bytes
    - LLaMA2-Chat
        - Type: Autoregressive Transformer model trained with a causal language modeling objective
        - Architecture
            - Analog to LLaMA 2
            - Ghost Attention (GAtt)
            - LLaMA2-Chat-SFT - Supervised Fine-Tuning 
                - Cosine learning rate schedule
                - LR: $2.0 \times 10^{-5}$
                - Weight Decay: 0.1
                - Batch Size: 64
                - Sequence Length: 4096
            - LLaMA2-Chat-RHLF - Reinforcement Learning with Human Feedback
                - Cosine learning rate schedule with LR down to 10% and 3% of the total number of steps as warmup (minimum 5)
                - LR: $1.0 \times 10^{-5}$ (7B and 13B) / $5.0 \times 10^{-6}$ (70B)
                - Batch Size: 512 pairs (1024 rows)
        - Parameters: 7B / 13B / 70B
        - Training datasets
            - SFT - Publicly available instruction tuning data, filtered and selected using annotations (27k annotations)
            - RHLF - Human-annotated comparisons from Anthropic Helpful/Harmless, OpenAI Summarize/WebGPT, StackExchange, Stanford SHP, Synthetic GPT-J and Meta (3M comparisons)
        - Languages: Primarily English
        - Notes
            - Obtained by fine-tuning LLaMA 2, optimizing it for dialogue use case
            - Supervised Fine-Tuning samples are prompt (with zeroed-out loss) + answer (separated by a special token)
            - RLHF Fine-Tuning samples are prompt + pair of model answers with human annotation and evaluation between the two
            - RHLF Fine-Tuning is further optimized using Iterative Fine-Tuning following [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) / [Rejection Sampling](https://arxiv.org/abs/2309.06657)
            

### [Textbooks Are All You Need II: phi-1.5 technical report](https://arxiv.org/abs/2309.05463)
- [Local Copy](PDFs/Interest/textbooks_are_all_you_need_ii_phi_1.5_technical_report.pdf)
- Model cards
    - phi-1.5
        - Type:
        - Architecture
            - Configuration and parameters based on [phi-1](https://arxiv.org/abs/2306.11644)
                - [Transformer model](https://arxiv.org/abs/1706.03762) with 24 layers and 32 attention heads of size 64
                - Context Length: 2k
                - [Rotary positional embeddings](https://arxiv.org/abs/2104.09864) of dimension 32
                - [FlashAttention](https://arxiv.org/abs/2205.14135)
                - [Codegen-mono tokenizer](https://arxiv.org/abs/2203.13474)
            - Constant learning rate
            - LR: $2 \times 10^{-4}$
            - Weight Decay: 0.1
            - Adam optimizer with $\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-7}$
            - Fp16 weights with [DeepSpeed ZeRO Stage 2](https://arxiv.org/abs/1910.02054)
            - Batch Size: 2048
        - Parameters: 1.3B
        - Training datasets
            - phi-1 training data (7B tokens) 80%
            - Synthetic textbook-like data on 20K topics (20B tokens) 20%
        - Languages: English
        - Notes
            - Focus on common-sense reasoning
            - Total training for 150B tokens
    - phi-1.5-web
        - Type:
        - Architecture
        - Parameters: 1.3B
        - Training datasets
            - Filtered web data from Falcon refined dataset, The Stack and StackOverflow (90B tokens) 40%
            - phi-1 training data (7B tokens) 20%
            - Synthetic textbook-like data on 20K topics (20B tokens) 40%
        - Languages: Primarily English
        - Notes
            - Web-enhanced version of phi-1.5
            - Total training for 100B tokens
            - Additional web-only version that is only trained on the filtered web data
            - No additional fine-tuning or RLHF
            - Attenuating effect of textbook-like data on toxic content generation