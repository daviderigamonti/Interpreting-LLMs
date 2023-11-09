# Thesis resources


## Papers


### Direct

#### Of Non-Linearity and Commutativity in BERT

*Resources:*
    [Remote](https://arxiv.org/abs/2101.04547) |
    [Local](PDFs/of_non-linearity_and_commutativity_in_bert.pdf) |
    [Annotated](PDFs/Annotated/of_non-linearity_and_commutativity_in_bert_annotated.pdf)

*Relevance:* ★★★☆☆

The paper introduces a tool to measure the non-linearity of different components in transformers and deep neural networks in general.
The authors make use of the linearity score $\gamma_f$ which measures "how much a certain component $f$ is linear", to do so they apply cosine similarity in the following way: $\gamma_f = \mathbb{E}_{\bm{e} \sim E}[cos(\bm{y}, \bm{y^*})]$, where $\bm{e} \sim E$ is an input sampled over a set of inputs, $\bm{y} = f(\bm{e})$ is the actual output and $\bm{y^*} = f^*(\bm{e})$ is the approximated output obtained by the linear approximator $f^*$ (which is trained to maximize the cosine similarity).

To apply this to transformers they consider the geometry of the embedding space by calculating the cone size for the embedding space at layer $l$ as: $ConeSize(l) = \mathbb{E}_{(\bm{e_i^l}, \bm{e_j^l}) \sim \bm{E}^l}[cos(\bm{e_i^l}, \bm{e_j^l})]$ where $\bm{E^l}$ are the embeddings at layer $l$.
They then use the $ConeSize$ to apply a correction to the linearity score and obtain the normalized linearity score $\tilde \gamma_f^l$.

By evaluating the normalized linearity score on MLP/SA-FF layers of BERT and experimenting with the removal of layers the authors discover that:
- MLPs introduce strong non-linearity, especially in the middle layers.
- The non-linearity introduced by SA-FF follows the same trend as that of MLPs even if the non-linear mechanism is fundamentally different.
- Replacing the MLPs by linear approximators yields approximately the same performance as completely removing the MLPs. (The non-linear contribution of MLPs is responsible for performance difference).
- The drop in performance after removing MLP blocks is smaller than when removing SA-FF blocks.
- Removing SA-FF is as harmful as removing the whole encoder layer.

The authors, then experiment with swapping 2 layers, shuffling all layers and weight sharing configurations, discovering that:
-  BERT is surprisingly robust to this layer swapping, and only 3 epochs of further fine-tuning can recover most of the performance loss.
- Swapping over longer distances generally results in worse performance, indicating that adjacent layers are more similar (not in the case of very first and last layers since they are remarkably similar, potentially due to their "closeness" to the input/output space).
- With shuffling the results show that as they decrease the number of fixed layers, performance monotonically decrease but with 3 epoch fine-tuning much of the performance can be recovered.
- Since the best layers for weight sharing are around 2-5 they notice that BERT has an inherent tendency towards weight-sharing, but its layers do still exhibit a certain hierarchical structure.

Their findings suggest that feature extraction in BERT is not strictly hierarchical, but happens incrementally, and that it is not the "early layers" per-se that extract low-level features, but low-level features are simply extracted first, regardless of the order in which the layers are applied.

#### How to Do a Vocab Swap? A Study of Embedding Replacement for Pre-trained Transformers

*Resource:* 
    [Remote](https://openreview.net/forum?id=MsjB2ohCJO1) |
    [Local](PDFs/how_to_do_a_vocab_swap_a_study_of_embedding_replacement_for_pre-trained_transformers.pdf) |
    [Annotated](PDFs/Annotated/how_to_do_a_vocab_swap_a_study_of_embedding_replacement_for_pre-trained_transformers_annotated.pdf)

*Relevance:* ★★★★★

This paper explores the possibility of swapping the vocabulary of a language model, particularly BERT, to adapt it to a different language or tokenizer, instead of adapting the language model "head" to the downstream task.
The authors investigate various initialization strategies and their effects on the quality of the vocabulary swap.

For most of the tasks they use the `bert-base-uncased`, `roberta-base`, `dbmdz/bert-base-german-cased`, `laubert/flaubert_base_cased` models as source models and MNLI, MRPC, STSB, XNLI and PAWX-S as downstream tasks.
BERT models are pre-trained with an MLM (Masked Language Modeling) objective, and fine-tuned with a task-specific head.
In some experiments all the weights are trained (All Grad), while in others the backbone is frozen and only the parameters of the embedding layer are trained (Embed Grade) and in some other cases only the embedding matrix is considered, freezing the positional embedding as well.
In some experiments all the backbone components of the transformer model are carried through to the target model, other times only the word embedding matrix utilizing the following techniques.

The main focus of the paper is the re-initialization of the word embeddings matrix and the proposed effectiveness of some *strong methods* to perform this operation using various *matching schemes* that map targets from the source vocabulary to the target vocabulary.
The paper considers the following *matching schemes*:
- *From Scratch [Weak]:* Entirely re-train the model from scratch, applying standard weight initialization and replacing the embedding matrix with a new one.
- *Reinit Embed [Weak]:* (Reinitialized Embedding + Pretrained Backbone) Replace only the embedding matrix while keeping the downstream language logic learned by the transformer; assuming that the function learned by the transformer blocks is general purpose and can be transferred to the target modeling problem.
- *Freq Match [Strong]:* (Frequency Matching) Initialize $W_{src}$ using vectors from the frequency ordered embedding $W_{tgt}$ as $w_{tgt}^{[i]} \leftarrow~w_{src}^{[i]}$ exploiting the implicit frequency ordering of the WordPiece algorithm.
However, the explicit frequency matching can be adopted by independently computing the occurrence counts
for each word in the vocabulary over the training set in each of the respective language and then re-indexing the vocabularies.
- *Align Match [Strong]:* (Alignment Matching) Two-stage method that, in the first phase, sees the use of fastText as a source of pretrained word vectors in the source and target natural language, the word vectors are then aligned using [this method](https://aclanthology.org/D18-1330.pdf).
During the second stage, each vector in the target language is assigned a group of nearest neighbors from the source language and the entries of $W_{tgt}$ are initialized as linear combinations of the vectors in $W_{src}$ as $w_{tgt}^{[i]} \leftarrow~\sum_j{\alpha_jw_{src}^{[j]}}$, $\sum_j{\alpha_j}=1$, where the mixing parameters $\{\alpha_j\}$ are largest for the closest neighbors. **[???]**
- *Dict Match [Strong]:* (Dictionary Matching) Use an additional single-word dictionary (generated through other translation means) to initialize vectors in the target embedding.

The paper analyzes 3 main experimental setups:
- *BERT Embedding Recovery:* Starting with a fully pretrained BERT checkpoint, they perturb (by scrambling the vectors or randomizing a percentage of them) the embedding and then continue training to see how accurately the original embedding is recovered.
- *FrankenBERTa:* They start with a pretrained `roberta-base` model that uses BPE vocabulary and attempt to swap it with the tokenizer of a `bert-base-uncased` model using the WordPiece algorithm.
- *Cross Lingual Transfer (CLT):* Realistic scenario of retrofitting a pretrained model to process a completely different natural language (either German or French).

We observe that, for toy embedding recovery problems, random initialization of embeddings generally performs poorly, while stronger initializations result in an "anchoring" effect that enables fairly accurate recovery of the original embedding.
Interestingly, this lesson carried over to the cross-lingual setting.
There too, we see that anchored initializations facilitate effective learning of embeddings.

Evaluations have been done considering *Embedding Correlation* between two word embedding matrices $W_1$ and $W_2$, both indexed by the token ids of the same vocabulary $V$, is defined as the average Pearson correlation of corresponding word vectors from the two embeddings: $\bar \rho(W_1, W_2) = \frac{1}{|V|}\sum{\rho_i}$, $\rho_i = \frac{\text{COV}(x_i, y_i)}{\sigma_x \sigma_y}$, $\forall i \in V$.

#### Combining pre-trained language models and structured knowledge

*Resources:* 
    [Remote](https://arxiv.org/abs/2101.12294) |
    [Local](PDFs/combining_pre-trained_language_models_and_structured_knowledge.pdf) |
    [Annotated](PDFs/Annotated/combining_pre-trained_language_models_and_structured_knowledge_annotated.pdf)

*Relevance:* ★★☆☆☆

The paper takes a wide overview on state-of-the-art techniques for combining structured knowledge with pre-trained language models.
It takes a schematic approach and doesn't delve into the implementation specifics of each model, giving a brief overview on the structure of the model, its accomplishments, its faults and how it could be improved.

Structured knowledge is usually interpreted as a Knowledge Graph (KG) which consists of a triple $G := (V, E, L)$ where $V$ is the set of vertices (concepts), $E \subseteq V \times L \times V$ is the set of edges (assertions) that may have weights and $L$ is the set of labels (relations).

Four main approaches for "injecting" the KG information into a language model:
- *Input focused injections:* Modify data during pre-processing o in pre-transformer layers, usually by converting assertion in a set of words and fine-tuning a pre-trained LM with these inputs.
However, it can also involve the modification of tokenization or embedding operations.
    - *AMS (Align Mask Select):* Automated pre-training approach which constructs a QA dataset aligned to a KG, utilizes graph-based confounders in generated dataset entries.
    - *COMET (COMmmonsEnse Transformers):* Generative GPT-based language model that can provide natural language representation of triple, useful for zero-shot KG completion and pre-processing of triples for training is relatively simple.
- *Architecture injections:* Approaches that involve the addition of additional layers to the model in order to integrate knowledge with contextual representation, or the modification of existing layers by manipulating attention mechanisms.
Use of adapter-like mechanisms to inject information into the model.
    - *KnowBERT:* Fusion of contextual and graph representation of entities, attention-enhanced entity-spanned knowledge infusion and permits the injection of multiple KGs in varying levels of the model.
    - *"Common sense or world knowledge?...":* Adapter based approach which fine-tunes a minimal amount of parameters and shows that a relatively small amount of additional iteration can inject the knowledge in the adapter, plus, adapters that are trained on KGs do indeed boost the semantic performance of transformer-based models.
- *Output injections:* Approaches that focus on changing either the output structure or the losses that were used in the base model in some way, in order to incorporate knowledge.
    - *SemBERT:* Encodes semantic roles in an entity embedding that is combined at the output.
- *Combination/Hybrid injections:* Approaches that perform any combination of input (I), architecture (A), output (O) injections or are a hybrid between Language Models (LM), Graph Reasoning (GR) and GCN Embeddings (GCN).
    - *KALM (Knowledge-Aware Language Model) [IO]:* Sends an entity signal in the beginning and enforces it in the output of a generative model to notice its semantics.
    - *Exploiting structured knowledge in text... [IO]:* Similar to KALM but with MLM objective, filters relevant entities to incorporate their information into the model and enforces entity signal at the beginning and end of the model through masking and max-margin losses.
    - *LIBERT (Lexically Informed BERT) [IO]:* Incorporates lexical constraints from entity embeddings and shows a good performance with constrained amounts of data.
    - *KG-BERT [IO]:* Fine-tunes BERT into completing triples from a KG, uses binary classification to predict if a triple is valid and utilizes multi-class classification to predict the relation type.
    - *CCCC (Cracking the Contextual Commonsense Code...) [IO]:* Notices that BERT has some commonsense information in some areas, but is lacking in others and tries to fine-tune it on the deficient areas to discover an increase in performance; finally, it points out that the combination of graph embeddings plus contextual representations is useful.
    - *ERNIE 2.0 [IO]:* A framework that permits flexibility on the underlying model, with a continual learning platform that keeps training older tasks to maintain their information and a wider variety of semantic pre-training tasks.
    - *K-BERT [IA]:* Utilizes attention mechanisms to mimic connected subgraphs of injected triples and also performs injection of relevant triples as text inputs.
    - *BERT-MK [AO]:* Utilizes a modified attention mechanism to mimic KG structure between terms and incorporates a triple reconstruction loss to train the KG-transformer modules that are merged with the regular transformer for a contextual + knowledge-informed representation.
    - *K-Adapter [AO]:* Approach that provides a framework for continual learning and uses a fusion of trained adapter outputs for evaluation tasks.
    - *Graph-based reasoning... [LM + GCN + GR]:* Combination of GCN, LM and search systems to answer questions, uses an XLNet as contextual embedding for GCN nodes and performs QA reasoning with the GCN output.
    - *Commonsense knowledge base completion... [LM + GCN]:* Uses a GCN and a LM to generate contextualized assertions representation by utilizing BERT to generate contextual embeddings for nodes and an encoder-decoder structure to learn triples.

### Cited

#### Bridging the data gap between children and large language models

*Paywall*

*Resources:* 
    [Remote](https://www.sciencedirect.com/science/article/abs/pii/S1364661323002036)
    
*Relevance:*

#### LoRA: Low Rank Adaptation of Large Language Models

*Resources:*
    [Remote](https://arxiv.org/abs/2106.09685) |
    [Local](PDFs/Cited/lora_low-rank_adaptation_of_large_language_models.pdf) !
    [Annotated](PDFs/Cited/Annotated/lora_low-rank_adaptation_of_large_language_models_annotated.pdf)

*Relevance:* ★★☆☆☆

The paper described a technique to fine-tune models by creating a more compact and specialized version of the weights, in this way a pre-trained model can be shared and used to build many versions for different tasks.
This approach works due to the intrinsic overparametrization of deep learning models, especially LLMs.
The author hypothesizes that the change in weights during model adaptation has a low "intrinsic rank", leading to the so called Low-Rank Adaptation (LoRA).

The main difference with other existing "model compression" techniques (such as adapters and input layer activation optimization) is that LoRA doesn't introduce any inference latency, can be seamlessly parallelized, is relatively easy to optimize and doesn't hinder the available sequence length.

Given a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA constraints its updates by representing it with a low rank decomposition: $W_0 + \Delta W = W_0 + BA$ where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$.
For this to work the rank $r$ should be such that $r \ll min(d,k)$ and $W_0$ should be kept completely frozen during training, so that only $\Delta W = BA$ is learned.

$A$ is initialized using a Random Gaussian and $B$ is initialized with zeros.
$\Delta W$ is scaled by a factor of $\frac{\alpha}{r}$ where $\alpha$ acts similarly to a learning rate.

The paper is limited to the analysis of LoRA's effect on attention weights for specific downstream tasks.
The author performs several empirical experiments to compare different baselines on a selection of different datasets and tasks; the baselines are:
- *Fine-Tuning (FT):* Classic adaptation approach, there are two main versions that can either have the whole model as trainable, or just a selection of the last layers.
- *Bias-only or BitFit:* Only bias vectors are trained, everything else is frozen.
- *Prefix-embedding tuning (PreEmbed):* Insert special tokens that have trainable word embeddings among the input tokens.
- *Prefix-layer tuning (PreLayer):* Like *PreEmbed*, but also activations are learned after every Transformer layer.
- *Adapter tuning:* Insert low-dimensionality adapter layers between self-attention and the subsequent residual connections.
- *LoRA:* Add trainable pairs of rank decomposition matrices in parallel to existing weight matrices.

From the empirical experiments, LoRA seems to match or exceed the fine-tuning performance on all tasks.

The author concludes the paper with some questions regarding Low-Rank Update and makes the following statements after empirically demonstrating their veridicity:
- Adapting (with LoRA) some specific attention matrices simultaneously ($\Delta W_q$ and $\Delta W_v$) is likely to yield the best results.
- $\Delta W$ matrices have a very small "intrinsic rank" as, even with small values for $r$, LoRA performs competitively.
- $\Delta W$ has a stronger correlation with $W$ w.r.t. a random matrix, meaning that $\Delta W$ amplifies some of the features that are already in $W$, and those features happen to be the ones that are relevant for the specific downstream task.

#### Cyclical Learning Rates for Training Neural Networks

*Resources:*
    [Remote](https://arxiv.org/abs/1506.01186) |
    [Local](PDFs/Cited/cyclical_learning_rates_for_training_neural_networks.pdf) | 
    [Annotated](PDFs/Cited/Annotated/cyclical_learning_rates_for_training_neural_networks_annotated.pdf)

*Relevance:* ★☆☆☆☆

The paper presents a peculiar learning rate schedule that presents a cyclical pattern.
The author compares it with other state-of-the-art learning rate schedules, expands on its theoretical significance and empirically demonstrates its qualities.

It is stated that conventional wisdom dictates that the learning rate should be a single value that monotonically decreases during training.
However, varying the learning rate during training can be beneficial, and doing so in a cyclical manner doesn't require any additional computation.
In addition, varying the learning rate also eliminates the need to tune it specifically.

The appeal of Cyclical Learning Rates (CLRs) is that increasing the learning rate, despite having a short-term negative effect on accuracy, also has a long-term beneficial effect for generalization.
Intuitively, CLRs work due to their ability of rapid traversal of saddle point plateaus, which pose the most difficulty in loss minimization.

The author experimented with numerous shape variations such as Triangular Window (linear), Welch Window (parabolic) and Hann Window (sinusoidal), but all shape produced similar results.

To identify a good range for CLRs, the author proposes a "LR range test", which consists of running the model for several epochs while letting the LR increase linearly between two arbitrary values that are distant enough.
Then, the accuracy-LR plot needs to be analyzed in order to find the lower bound as the LR for which the accuracy starts to significantly increase, and the upper bound as the LR for which the accuracy slows, becomes ragged or drops.

The author performs several experiments using various architectures, on a plethora of tasks and finds that employing CLR almost always poses a significant improvement in accuracy w.r.t. fixed LR and adaptive LR.

#### Universal Language Model Fine-tuning for Text Classification

*Resources:*
    [Remote](https://aclanthology.org/P18-1031/) |
    [Local](PDfs/Cited/universal_language_model_fine-tuning_for_text_classification.pdf) |
    [Annotated](PDfs/Cited/Annotated/universal_language_model_fine-tuning_for_text_classification_annotated.pdf) |

*Relevance:* ★☆☆☆☆

This paper presents a universal method to fine-tune language models with the main purpose of text classification.
The rationale behind this idea is following the steps taken in Computer Vision (CV) deep learning models, for which inductive transfer learning has had a large impact, especially paired with the notorious ImageNet dataset.
The author believes that with proper pre-training and fine-tuning techniques, it would be possible to replicate the advancements made in CV, to the NLP field.
This novel approach is then compared with other state-of-the-art fine-tuning methods.

The chosen source task for pre-training is Language Modeling since it captures many facets of language relevant for other downstream tasks.
The method is universal in the sense that works across documents with varying length, syntax, label and number, it uses a single architecture and training process, it requires no custom feature engineering or preprocessing, and it does not require additional in-domain documents or labels.
The author proposes a variety of novelties included in the Universal Language Model FineTuning (ULMFiT):
- *Discriminative fine-tuning:* As different layers capture different types of information they should be fine-tuned to different extents.
To do so, the SGD update formula becomes: $\theta_t^l = \theta_{t-1}^l - \eta^l \cdot \nabla_{\theta^l}J(\theta)$ where $\theta = \{\theta^1, ..., \theta^L\}$ are the model parameters for each of the $L$ layers, $\eta = \{ \eta^1, ..., \eta^L\}$ are the learning rates for each of the $L$ layers and $\nabla_{\theta^l}J(\theta)$ is the gradient for the current layer $l \in L$.
- *Slanted traingular learning rates (STLR):* To incentivize the model to quickly converge to a suitable region of the parameter space in the beginning of the trained it's possible to use this technique which first increases the LR and the linearly decays it following an update schedule.
The key for good performance is inside the short increase and long decay period.
- *Gradual unfreezing:* To avoid catastrophic forgetting by fine-tuning all the layers at the same time, it's possible to progressively unfreeze each layer for each epoch starting from the last one.
- *BackPropagation Through Time for Text Classification (BPT3C):* LMs are usually trained with BPTT, however, to make fine-tuning feasible it's possible to divide a document into fixed-length batches of size $b$.
At the beginning of each batch, the model is initialized with the final state of the previous batch and training is repeated, keeping track of all the relevant information to be applied at the end of the set of batches.

The actual models that are used are a variety of LSTM-based models (both unidirectional and bidirectional variants) which do not include attention.

The author selects 6 datasets over 3 text classification tasks (sentiment analysis, question classification and topic classification) to empirically evaluate ULMFiT against other fine-tuning baselines and other "vanilla" baselines.
For all the identified tasks, the reported results are always positive or equal in favor of the ULMFiT w.r.t other baselines.
The bidirectional variants of the various models consistently outperform the unidirectional relatives.

#### An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks

*Resources:*
    [Remote](https://arxiv.org/abs/1312.6211) |
    [Local](PDFs/Cited/an_empirical_investigation_of_catastrophic_forgetting_in_gradient-based_neural_networks.pdf)

*Relevance:* ★★☆☆☆

This paper discusses the topic of catastrophic forgetting and its nuances in the world of neural networks.
When a learning system is first trained on one task, then trained on a second task, it may forget how to perform the first one.
For once, it is not a neural network-specific problem, as it has been observed in other learning systems and even biological memory systems.

The goal of the author is to shed light on this phenomenon and empirically experiment with particular techniques in order to find a way to mitigate the problem.
The basic algorithms and approaches used in the experiments are:
- *Dropout:* Designed to regularize a neural network (to improve generalization) by multiplying the weights of a layer with a binary mask.
During training time, the binary mask is randomized by sampling each element using a fixed probability $p$; this happens for each singular example shown to the model.
At inference time the mask is set to all 1s and units are multiplied by $p$ to compensate.
Dropout is an effective regularizer and very efficient, thus it enables training of noticeably larger networks.
- *Activation functions:* The output of each neuron in a given layer is calculated by computing the presynaptic activation $z = Wx + b$ and then feeding it through an activation function $f$, obtaining $h = f(z)$.
There are a variety of different activation functions, however the ones taken into consideration in this paper are:
    - *Logistic sigmoid:* $f(z)_i = \frac{1}{1+e^{-z_i}}$.
    - *Rectified linear unit:* $f(z)_i = max(0, z_i)$.
    - *Hard Local Winner Take All (LWTA):* $f(z)_i = g(i, z)z_i$ where $g$ is a gating function equal to 1 if $z_i$ is the maximal element of its block ($z$ is divided into $k$ disjointed blocks), 0 otherwise.
    - *Maxout:* $f(z)_i = max_j\{z_{ki},...,z_{k(i+1)-1}\}$.
- *Random hyperparameter search:* The performance of most deep learning methods is a complicated non-linear function of multiple hyperparameters.
Automated selection of hyperparameters allows for fair comparison, but is challenging.
Grid search is suffers from the curse of dimensionality and its time requirements can easily grow exponentially.
Bayesian optimization and other methods are more sophisticated and efficient but are not trivial to implement and might introduce bias.
Random hyperparameters search is simple to implement and relatively low-cost, however it can still obtain roughly state-of-the-art results.

For each experiment 2 tasks are defined, an "old" one and a "new" one.
The goal is to examine the behavior of models that are initially trained on the old task and then subsequently trained on the new one.
The author defines 3 definition of old-new task couples:
- *Input Reformatting:* The tasks are functionally identical but with different input formats.
- *Similar tasks:* The tasks are similar.
- *Dissimilar tasks:* The tasks are dissimilar.

Before delving into the evaluation results, the author takes the time to specify how each layer and component has been initialized in order to produce results that are as comparable and reproducible as possible.
In addition, models are trained on the "old task" until the validation set error has not improved in the last 100 epochs and on the "new task" until the union of old and new validation sets has not improved in the last 100 epochs.
All parameters are restored to the epoch with the best validation performance.

From the experiment results, it is empirically proven that training with dropout is always beneficial and overall it helps to prevent forgetting.
That said, there are some unexplained properties that caused dropout to preserve the dimension of the model in task 1 and 2, but decrease it for task 3.
It was also found that, apart some exceptional cases (LWTA), the activation function does not influence the issue of catastrophic forgetting and its performance in a given task is mainly problem dependent.

One interesting fact that happened during the first task, by visualizing the weights of the first layer of the resulting network, the author discovered that the networks didn't learn to map different input formats intro pre-existing concepts, but rather the higher layers adapted to accommodate a relatively arbitrary projection of the input.

#### When and Why are Pre-trained Word Embeddings Useful for Neural Machine Translation?

*Resources:*
    [Remote](https://arxiv.org/abs/1804.06323) |
    [Local](PDFs/Cited/when_and_why_are_pre-trained_word_embeddings_useful_for_neural_machine_translation.pdf) |
    [Annotated](PDFs/Cited/Annotated/when_and_why_are_pre-trained_word_embeddings_useful_for_neural_machine_translation_annotated.pdf)

*Relevance:* ★★☆☆☆

The paper is aimed at exploring the value of pre-trained word embeddings in Neural Machine Translation(NMT) tasks.
The author utilizes data between English (EN) and three pairs of languages, where the two languages in the pair are similar, with one being relatively low-resourced compared to the other: Galician (GL) and Portuguese (PT), Azerbaijani (AZ) and Turkish (TR), and Belarusian (BE) and Russian (RU).

Five main questions are revealed and tackled throughout the paper:
- *Q1:* Is the behavior of pre-training affected by language families and other linguistic features of source and target languages?
- *Q2:* Do pre-trained embeddings help more when the size of the training data is small? 
- *Q3:* How much does the similarity of the source and target languages affect the efficacy of using pre-trained embeddings?
- *Q4:* Is it helpful to align the embedding spaces be-
tween the source and target languages?
- *Q5:* Do pre-trained embeddings help more in multilingual systems as compared to bilingual systems?

The author explores this outlets by performing experiments using a standard 1-layer encoder-decoder architecture with attention and exploiting language-specific FastText embeddings of size 300 trained on a Wikipedia corpus.

The main giveaways for each question are the following:
- *Q1:* The majority of the gain from pre-trained word embeddings results from a better encoding of the source sentence.
- *Q2:* The gain is highest when the baseline system is poor but not too poor, this suggests that at least a moderately effective system is necessary before pre-training takes effect.
- *Q3:* If the two languages in the translation pair are more linguistically similar, the semantic neighborhoods will be more similar between the two languages.
As a result, we may expect that the gain from pre-training of embeddings may be larger when the source and target languages are more similar.
- *Q4:* The alignment of word embeddings was not beneficial for training, with gains or losses essentially being insignificant across all languages.
"A priori" alignment of embeddings may not be necessary in the context of NMT, since the NMT system can already learn a reasonable projection of word embeddings during its normal training process.
- *Q5:* For multilingual systems both pre-training and alignment ensure that the word embeddings of the two source languages are put into similar vector spaces, allowing the model to learn in a similar fashion as it would if training on a single language.

#### Learned in Translation: Contextualized Word Vectors

*Resources:*
    [Remote](https://proceedings.neurips.cc/paper_files/paper/2017/file/20c86a628232a67e7bd46f76fba7ce12-Paper.pdf) |
    [Local](PDFs/Cited/learned_in_translation_contextualized_word_vectors.pdf)

*Relevance:* ★★★★☆

The paper in question introduces the CoVe (Context Vectors) approach as a particular way to transpose an embedding layer from a model to another.
The author asserts that machine translation is the best suitable source domain for pre-training the model due to the quantity and quality of information that needs to be encompassed by the encoder module.

In the first step, the paper describes an English-to-German translation given a sequence of words in the source language $w^x = [w_1^x,...,w_n^x]$ and a sequence of words in the target language $w^z = [w_1^z,...,w_m^z]$.
The first sequence is transformed in GloVe embeddings first, then the resulting vectors are fed to a standard, two-layer bi-LSTM network referred as MT-LSTM resulting in a sequence of hidden states $h =~\text{MT-LSTM}(GloVe(w^x))$.

At time-step $t$, the decoder uses a two-layer, unidirectional LSTM to produce a hidden state $h_t^{dec}$ based on the previous decoder hidden state $h_{t-1}^{dec}$, previous target embeddings $z_{t-1}$ obtained from target word vector $w^z$ and the previous context-adjusted hidden state $\tilde h_{t-1}$ resulting in: $h_t^{dec} =~\text{LSTM}([z_{t-1};\tilde h_{t-1}], h_{t-1}^{dec})$.

From the hidden encoder states stacked along the time dimension $h \rightarrow H$ and the hidden decoder state it's possible to obtain a vector of attention weights $\alpha_t =~\text{softmax}(H(W_1h_t^{dec}+b_1))$ representing the relevance of each encoding time-step to the current decoder state.
These attention weights are used as coefficients in an attentional sum that is then concatenated with the hidden decoder state to obtain the context-adjusted hidden state $\tilde h_t =~tanh([W_2H^\top\alpha_t + b_2; h_t^{dec}])$.

Finally, the distribution over output words in generated by applying a softmax $p(\hat w_t^x|X, w_1^z,...,w_{t-1}^z) =~\text{softmax}(W_{out}\tilde h_t + b_{out})$.

In the second step, we are able to transfer the knowledge that has been learned by the MT-LSTM to downstream tasks by treating its outputs as context vectors $CoVe(w) =~\text{MT-LSTM(GloVe(w))}$ that can be concatenated with the original embedding: $\tilde w =~[GloVe(w); CoVe(w)]$.

The paper proceeds to describe a possible target architecture used for classification and question answering tasks.
The target architecture is a general Biattentive Classification Network (BCN) that sees a ReLU + LSTM encoding layer, a biattention mechanism integrated by two LSTMs, timewise pooling layers and a 3-layer maxout network.

As far as results go for the majority of tasks and experiments, model that use CoVe alongside GloVe embeddings seems to outperform models that only use GloVe.
In addition, there appears to be a positive correlation between larger NMT datasets (containing more complex and varied language) and the improvement that CoVe brings to downstream tasks.

The paper concludes by illustrating the differences between the CoVe approach and Skip-Though Vectors (Kiros et al. 2015).

#### A Structural Probe for Finding Syntax in Word Representations

*Resources:*
    [Blog](https://nlp.stanford.edu//~johnhew//structural-probe.html?utm_source=quora&utm_medium=referral#the-structural-probe) |
    [Remote](https://nlp.stanford.edu/pubs/hewitt2019structural.pdf) |
    [Local](PDFs/Cited/a_structural_probe_for_finding_syntax_in_word_representations.pdf) |
    [Annotated](PDFs/Cited/Annotated/a_structural_probe_for_finding_syntax_in_word_representations_annotated.pdf)

*Relevance:* ★★☆☆☆

The paper presents a structural probing method to test if syntax trees are consistently embedded in a linear transformation of a neural network’s word representation space.
The probe uses supervision to find the transformation under which the wanted properties are best approximated for each model.

Given $\mathcal{M}$ model that takes in a sequence of $n$ words $w_{1:n}^\ell$ and produces a sequence of vector representations $\bm h_{1:n}^\ell$, where $\ell$ identifies the sentence.
It's possible to define a family of squared distances as $d_B(\bm h_i^\ell, \bm h_j^\ell)^2 =(B(\bm h_i^\ell - \bm h_j^\ell))^\top(B(\bm h_i^\ell - \bm h_j^\ell))$ where $i$ and $j$ index the words in the sentence and $B$ is the parameter matrix that is learned through gradient descent $min_B{\sum_\ell{\frac{1}{|s^\ell|^2}\sum_{i,j}{|d_{T^\ell}(w_i^\ell, w_j^\ell) - d_B(\bm h_i^\ell, \bm h_j^\ell)^2|}}}$ having $T^\ell$ representing all sentences in the training set and $|s^\ell|$ the length of given sentence.

The defined probe tests the concrete claim that there exists an inner product on the representation space whose squared distance encodes syntax tree distance.
This means that the model not only encodes which word is governed by which other word, but each word’s proximity to every other word in the syntax tree.
In addition, the paper also defines a structural probe for tree depth.

The author performs some experiments on various ELMo and BERT architectures utilizing some evaluation metrics.
For the first task, tree reconstruction is evaluated by taking each test sentence’s predicted parse tree distances, computing the minimum spanning tree and evaluating the predicted tree on undirected attachment score (UUAS) (corresponding to the percent of undirected edges placed correctly) against the gold tree.
Distance correlation is also taken into consideration as Spearman correlation is computed between true and predicted distances for each word in each sentence and then averaged across all sentences of fixed length. 

For the second task, Spearman correlation between the true depth ordering and the predicted ordering is used, averaging between sentences of the same length.
Additionally, models’ ability to identify the root of the sentence as the least deep is also considered.

#### Visualizing and Measuring the Geometry of BERT

*Resources:*
    [Blog](https://pair-code.github.io/interpretability/bert-tree/) |
    [Remote](https://arxiv.org/pdf/1906.02715.pdf) |
    [Local](PDFs/Cited/visualizing_and_measuring_the_geometry_of_bert.pdf) |
    [Annotated](PDFs/Cited/Annotated/visualizing_and_measuring_the_geometry_of_bert_annotated.pdf)

*Relevance:* ★★☆☆☆

The paper takes great inspiration from [this paper](#a-structural-probe-for-finding-syntax-in-word-representations), expanding upon it and giving answers for some of its questions.
Yet again, the main topic of this paper is studying the internal representation of knowledge inside a transformer model with the help of a probing approach.
This paper also gives some importance to BERT's attention layers, and it tries to extract semantic and syntactic information from the attention weight matrices as they are explicitly built on the relations between pairs of words.

To explore the attention matrices the author makes use of an attention probe (derived from edge probing techniques) which takes as input the model-wide attention vector that is formed by a concatenation of entries $a_{ij}$ from all the attention matrices given the token pair $(token_i, token_j)$.
This is to show that if a simple linear model is able to find a pattern in the model-wide attention vector, there exists a certain kind of dependency relation between two words encoded in it.

The author defines two of these probes as L2-regularized linear classifiers trained with SGD.
The first one is a simple linear binary classifier to predict whether an attention vector corresponds to the existence of a dependency relation between two tokens.
The second one is a multiclass classifier to predict which type of dependency relation exists between two tokens, given the dependency relation’s existence.
The success of these simple linear probes suggests that syntactic information is in fact encoded in the attention vectors.

The author demonstrates via mathematical proof, aided by the definition of power-p embedding, why parse tree distance seems to correspond specifically to the square of Euclidean distance, and whether some other metric might do better.
Exact power-2 embeddings (Euclidean embeddings) are then compared to BERT embeddings with a visualization tool based on PCA to reduce the embeddings dimension.
Through this analysis the authors discovers some systematic differences between the two representation, which suggest that BERT’s syntactic representation has an additional quantitative aspect beyond traditional dependency grammar.

A visual clustering analysis is done to unveil the way BERT's embeddings account for words that have multiple meanings.
Different senses of a word are typically spatially separated, and within the clusters there is often further structure related to fine shades of meaning.

To confirm this hypothesis, the author trains a probe to test the internal representation of BERT over a word-sense disambiguation (WSD) task.
Their results support that there is more semantic information in the geometry of earlier-layer embeddings and the idea that word sense information may be contained in a lower-dimensional space.

A last experiment is performed to test the hypothesis that if word sense is affected by context and encoded by location in space, it should be possible to influence context embedding positions by systematically varying their context.
This is confirmed by comparing individual sentences with a certain word and concatenated sentences that use the same word with different meanings.
For evaluation, the ratio of the cosine similarity between the embedding of a sentence and the centroid of its meaning and the centroid of its opposing meaning is used.

Results show how a token’s embedding in a sentence may systematically differ from the embedding for the same token in the same sentence concatenated with a non-sequitur.
This points to a potential failure mode for attention-based models: tokens do not necessarily respect semantic boundaries when attending to neighboring tokens, but rather indiscriminately absorb meaning from all neighbors.


## Articles

#### Can LLMs learn from a single example?

*Resources:* 
    [Post](https://www.fast.ai/posts/2023-09-04-learning-jumps/)

*Relevance:* ★★☆☆☆

*Relevant Citations:* 
    [Data Gap](#bridging-the-data-gap-between-children-and-large-language-models) |
    [LoRA](#lora-low-rank-adaptation-of-large-language-models) |
    [Cyclical Training](#cyclical-learning-rates-for-training-neural-networks) |
    [Universal Fine-tuning](#universal-language-model-fine-tuning-for-text-classification) |
    [Catastrophic Forgetting](#an-empirical-investigation-of-catastrophic-forgetting-in-gradient-based-neural-networks)

The article describes an experience made by the author while fine-tuning an LLM for a question answering task (QA) regarding multiple-choice science exam questions.

After training the model for 3 epochs on a dataset of questions, the author notices an odd loss curve.
The curve shows significant improvements at the start of each epoch but stable behavior throughout all epochs.
After analyzing their code for possible errors and suggesting the idea of a possible bug in the HuggingFace `Trainer` library, the author sets up an experiment to verify if the model isn't actually overfitting on a small subset of the training data.

By setting up a cyclical learning rate schedule for the training and repeating the training over 2 epochs they observed that:
- The loss curve for the first epoch looked very standard.
The learning rate was still warming up over the first 10% of the epoch, and then gradually decreased following a cosine schedule.
Once the LR came up to temperature, the training and validation loss rapidly decreased, and then they both slowed down.
- The first batches of the second epoch were when the learning rate was still warming up, therefore there is no immediate step-change.
Towards the end of that first 10% of the epoch, the training loss plummeted, because the LR was high when these batches were seen during the first epoch, and the model has learned what they looked like.
The model quickly learned that it could very confidently guess the correct answer.
But during this time, the validation loss suffered.
That’s because although the model is getting very confident, it’s not actually getting any better at making predictions since it has simply memorized the dataset.
Towards the end of the curve the training loss started to get worse and this made perfect sense under the memorization hypothesis as these are the batches that the model saw at a time when the LR had come back down again, so it wasn’t able to memorize them as effectively.
In the end, the model slowly recalibrated itself to a reasonable level of confidence.

From these findings, the author then tries to apply 1-cycle training over 3 epochs utilizing a single LR warm up at the start (10% of the batches) followed by a cyclical LR decay over the remaining batches.
The resulting behavior is similar, but the loss spike is delayed to the third epoch as it's dependent on the confidence of the model prediction, which is loosely mirrored by the loss.

The author concludes with a remark about how the validation loss doesn't actually matter for a model, but it is its accuracy that measures its actual performance.
It doesn't matter if the model is overconfident as long as it is right.

#### How embeddings learned from one model can be used in another?

*Resources:* 
    [Post](https://ai.stackexchange.com/questions/37542/how-embeddings-learned-from-one-model-can-be-used-in-another)

*Relevance:* ★★★☆☆

*Relevant Citations:* 
    [Word Embeddings](#when-and-why-are-pre-trained-word-embeddings-useful-for-neural-machine-translation)

The author(OP) of the StackExchange question asks how can embeddings be swapped between models without causing problems, since they sustain that embeddings should be model-specific.
OP brings a specific citation from [this article](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/) that says (regarding embeddings):

> [...] It can be used alone to learn a word embedding that can be saved and used in another model later.
> It can be used as part of a deep learning model where the embedding is learned along with the model itself.
> It can be used to load a pre-trained word embedding model, a type of transfer learning. [...]

The main focus of the cited article is to explain embeddings and apply them to toy problems using the Keras API, however its explanation is limited to contextualizing pre-trained embeddings into small-scale fully connected architectures and never mentions nor makes use of transformer architectures.

The responses to the original question are discordant, however the accepted answer mentions that, although embeddings may be trained with a specific model architecture, they are basically a linear layer and there is no reason to believe that they can't be applied to another model.
As it emerges from the discussion under the accepted answer, OP's main concern was about the congruence of output dimensions between the embedding layer and the input layer for the rest of the target model.
This is then clarified by pointing out that the output dimension of the pre-trained embeddings can be adapted depending on the needs of the model, alternatively, the target model needs to be built with the pre-trained embedding size in mind.

The accepted answer references an interesting paper regarding word embeddings usage.

#### Replacing word embeddings by contextualized word vectors

*Resources:*
    [Post]() |
    [Local](PDFs/replacing_your_word_embeddings_by_contextualized_word_vectors/index.md)

*Relevance:* ★★★☆☆

*Relevant Citations:* 
    [Contextualized Word Vectors](#learned-in-translation-contextualized-word-vectors)

The article outlines the design and architecture of contextualized word vectors, drawing from the machine translation (MT) framework.
Instead of traditional techniques like skip-gram or matrix factorization, the author leverages MT to create Contextualized Word Vectors (CoVe).

This involves training an encoder-decoder model for MT, assuming that MT can capture the essence of word meaning effectively and then "transferring" the encoder layer to use word vectors in other NLP tasks, such as classification and question answering.
By concatenating CoVe with traditional word embeddings like GloVe, this approach aims to provide richer and more context-aware word representations.
[CoVe in Pytorch] (Original)

The author links to a [Keras](https://github.com/rgsachin/CoVe) and a [PyTorch](https://github.com/salesforce/cove) implementation on GitHub (the PyTorch one being the original), referencing the original paper's idea.

#### Deconstructing BERT

*Resources:* 
    [Post](https://towardsdatascience.com/deconstructing-bert-reveals-clues-to-its-state-of-art-performance-in-nlp-tasks-76a7e828c0f1)
    
*Relevance:* ★★★☆☆

*Relevant Citations:*
    [Syntax in Word Representatons](#a-structural-probe-for-finding-syntax-in-word-representations) |
    [BERT Geometry](#visualizing-and-measuring-the-geometry-of-bert)

The article focuses on bringing light to the results obtained by two papers about the geometric insights of BERTS's embedding space.
Some of the key findings are:
- BERT’s word vector output encodes rich linguistic structure.
BERT approximately encodes syntax trees in the word embeddings it outputs for a sentence, and it is possible to recover these trees by a linear transformation of the word embeddings.
- BERT appears to encode syntactic and semantic features in word vectors in complementary subspaces.
- Different senses of a word have representations (determined by the sentence context) that are spatially separated in a fine-grained manner.

Words in a sentence are given locations in a high-dimensional space and if we subject these word vectors to a specific transformation, the Euclidean distance between these locations maps to syntax tree distance.
Thus, it's possible to recover the implicit syntax tree of a sentence by using specific linear transformations on its word vectors.
However, the syntax tree distance between two words corresponds to the square of the Euclidean distance between the corresponding nodes in the extracted minimum spanning tree.

The shape of the recovered tree is only approximately similar to ideal tree and the discrepancy has some patterns.
The average embedding distance between dependency relations varies widely, and it is not known what these differences mean.

It is also showed that some information about the syntax tree is also captured by the attention matrix as a liner classifier that took scalar values corresponding to bigrams of a sentence from all attention heads as inputs performed reasonably well at identifying eventual relationships between the words in the bigram.

By simply visualizing the embedding for words with multiple meanings in different sentence contexts, we can see how word sense affects embeddings.
Word sense disambiguation is accomplished by physically separating different semantics of a word inside the embedding space.


## Libraries

#### Parl.ai

*Resources:* 
    [Website](https://parl.ai/) |
    [PypI](https://pypi.org/project/parlai/)

*Relevance:* ★★★★☆

Parl-ai is a unified platform for sharing, training and evaluating dialogue models across many tasks.
It allows for integration with pytorch and supports versions of pytorch from 1.6 and higher.
Parl.ai exposes a wide range of popular datasets, pretrained models and reference models for computing baselines.

It's possible to swap out transformers' components by using [this feature](https://parl.ai/docs/tutorial_swap_components.html).
To do so, it's sufficient to add the `@swappable` decorator (with the target module that needs to be swapped as a parameter) to any class definition that inherits from the `nn.Module` class.

The swap needs to be performed **before** the model has been instantiated and the new component should have the same `__init__` and `forward` method signatures to function correctly.