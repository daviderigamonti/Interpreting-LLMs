# A Primer on the Inner Workings of Transformer-based Langauge Models (May 2024)

## Topics
Complete survey on transformer interpretability techniques.

## Abstract
The rapid progress of research aimed at interpreting the inner workings of advanced language models has highlighted a need for contextualizing the insights gained from years of work in this area.
This primer provides a concise technical introduction to the current techniques used to interpret the inner workings of Transformer-based language models, focusing on the generative decoder-only architecture.
We conclude by presenting a comprehensive overview of the known internal mechanisms implemented by these models, uncovering connections across popular approaches and active research directions in this area.

## Approaches

### Transformer Structure

Following recent literature regarding **interpretability** in Transformers, we present the architecture adopting the *residual stream* perspective.
In this view, each input *embedding* gets updated via vector additions from the *attention* and *feed-forward* blocks, producing *residual stream* states (or *intermediate* representations).
The final layer residual stream state is then projected into the vocabulary space via the *unembedding matrix* $W_U$, and *normalized* via the *softmax* function to obtain the probability distribution over the vocabulary from which a new token is sampled.

#### Layer Normalization

**Layer normalization** is a common operation used to *stabilize* the training process of deep neural networks.
Although early transformer models implemented *LayerNorm* at the *output* of each block, modern models consistently normalize *preceding* each block.

Given a representation $z$ we have:
$$LayerNorm(z) = \bigl(\frac{z-\mu z}{\sigma(z)}\bigr) \odot \gamma + \beta$$
Where $\mu$ and $\sigma$ calculate the *mean* and *standard deviation*, and $\gamma \in \mathbb{R}^d$ and $\beta \in \mathbb{R}^d$ refer to learned element-wise transformation and bias respectively.

We note that current LMs adopt an *alternative layer normalization* procedure, *RMSNorm*, where the centering operation is *removed*, and *scaling* is performed using the *root mean square* (RMS) statistic.

#### Attention

**Attention** is a key mechanism that allows transformers to *contextualize* token representations at each layer.
The *attention* block is composed of multiple *attention heads*.
At a decoding step $i$, each attention head reads from *residual streams* across previous positions ($\le i$), decides which positions to attend to, gathers information from those, and finally writes it to the current *residual stream*.

We have that *attention* at layer $l$ for token $i$ is given as:
$$Attn^l(X_{\le i}^{l-1}) = \sum_{h = 1}^{H}{Attn^{l,h}(X_{\le i}^{l-1})}$$
Where the single *head* $h$ contribution is given as:
$$Attn^{l,h}(X_{\le i}^{l-1}) = \sum_{j \le i}a_{i,j}^{l,h}x_j^{l-1}W_V^{l,h}W_O^{l,h} = \sum_{j \le i}a_{i,j}^{l,h}x_j^{l-1}W_{OV}^{l,h}$$
And where a single attention weight $a_{i,j}^{l,h}$ is given by:
$$a_{i,j}^{l,h} = softmax \Bigl(\frac{x_i^{l-1}W_Q^{l,h}(X_{\le i}^{l-1}W_K^{l,h})^\top}{\sqrt{d_k}}\Bigr) = softmax \Bigl(\frac{x_i^{l-1}W_{QK}^{l,h}X_{\le i}^{l-1}}{\sqrt{d_k}}\Bigr)$$
Notice that some matrices get combined into the *OV (output-value)* circuit $W_{OV}$ and *QK (query-key)* circuit $W_{QK}$ according to the rearrangement proposed by Kobayashi et al.

#### Feedforward Block

The **feedforward network** (**FFN**) in the transformer block is composed of two *learnable* *weight matrices* $W_{in}^{l}$ and $W_{out}^{l}$.
The residual $x_i^{mid,l}$ is multiplied by $W_{in}^{l}$, passed through an *element-wise non-linear activation function* $g$ and transformed by $W_{out}^{l}$ to produce the output $FFN(x_i^{mid,l})$, which is then added back to the residual stream:
$$FFN(x_i^{mid,l}) = g(x_i^{mid,l}W_{in}^l)W_{out}^l = \sum_{u=1}^{d_{ffn}}{g_u(x_i^{mid,l}w_{in_u}^l)w_{out_u}^l} = \sum_{u=1}^{d_{ffn}}{n_u^lw_{out_u}^l}$$

The computation was equated to *key-value memory retrieval* with keys $(w_{in}^l)$ acting as *pattern detectors* over the input sequence and values $(w_{out}^l)$ being *upweighted* by each neuron activation.
$n^l$ is the vector of *activations* with dimension $d_{ffn}$.

#### Prediction Head

The **prediction head** of a Transformer consists of an *unembedding matrix* $W_U$, sometimes accompanied by a *bias*.
The *last* residual stream state gets transformed by this linear map converting the representation into a next-token distribution of *logits*, which is turned into a *probability* distribution via the *softmax* function.

#### Residual View

The **residual stream** view shows that every model component interacts with it through *addition*.
Thus, the *unnormalized* scores (*logits*) are obtained via a *linear projection* of the summed component outputs.
Due to the properties of linear transformations, we can rearrange the traditional forward pass formulation so that each model component contributes *directly* to the output *logits*.

*Residual networks* also work as *ensembles* of *shallow* networks, where each subnetwork defines a path in the computational graph.
If we only consider attention we have two types of contributions:
- **Full OV circuits**, which only traverse a single OV matrix are written as $W_EW_{OV}W_U$
- **Virtual attention heads**, doing *V-composition* since the sequential writing and reading of two heads is seen as OV matrices composing together.
While, *Q-composition* and *K-composition* (compositions of $W_Q$ and $W_K$ with the $W_{OV}$ output of previous layers) can also be found in full transformer models.

### Behavior Localization

Understanding the inner workings of language models implies localizing which elements in the forward pass are *responsible* for a specific prediction.
Two different types of methods that allow localizing model behavior:
- **input attribution**
- **model component attribution**

#### Input Attribution

**Input attribution** methods are commonly used to *localize* model behavior by estimating the *contribution* of *input elements*.

Popular input attribution approaches were shown to be *insensitive* to *variations* in the model and data generating process, to *disagree* with each others' predictions and to show *limited capacity* in detecting
unseen *spurious correlations*.

- **Gradient-based input attribution**: For neural network models like LMs, *gradient information* is frequently used as a natural metric for *attribution* purposes.
*Gradient-based attribution* in this context involves a *first-order Taylor expansion* of a Transformer at a point $x$.
The resulting gradient captures intuitively the *sensitivity* of the model to each element in the input when predicting token $w$.
While attribution scores are computed for *every dimension* of input token embeddings, they are generally aggregated at a *token level* to obtain a more intuitive overview of the *influence* of *individual* tokens.
However, these approaches are known to exhibit *gradient saturation* and *shattering* issues.
This fact prompted the introduction of methods such as *integrated gradients* and *SmoothGrad* to filter *noisy* gradient information.
Finally, approaches based on *Layer-wise Relevance Propagation* (*LRP*) have been widely applied to study transformer-based LMs.
These methods use custom rules for gradient *propagation* to decompose component contributions at every layer, ensuring their sum remains *constant* throughout the network.
- **Perturbation-based input attribution**: estimate input importance by adding *noise* or *ablating input elements* and measuring the resulting *impact* on model *predictions*.
A multitude of *perturbation-based* attribution methods exist in the literature, such as those based on *interpretable local surrogate models* such as *LIME*, or those derived from *game theory* like *SHAP*.
Notably, new perturbation-based approaches were proposed to leverage *linguistic structures* and transformer components.
- **Context mixing for input attribution**: While raw model internals such as *attention weights* were generally considered to provide *unfaithful* explanations of model *behavior*, recent methods have proposed alternatives to *attention weights* for measuring *intermediate* *token-wise* attributions such as the use of the *norm* of *value-weighted* vectors and *output-value-weighted* vectors, or the use of vectors' *distances* to estimate contributions.
A common strategy among such approaches involves aggregating *intermediate per-layer attributions* reflecting *context mixing* patterns using techniques such as *attention rollout*, resulting in input attribution scores.
- **Contrastive input attribution**: An important limitation of input attribution methods for interpreting language models is that attributed output tokens belong to a *large* vocabulary space, often having *semantically equivalent* tokens competing for probability mass in next-word prediction. 
In this context, attribution scores are likely to *misrepresent* several *overlapping* factors such as grammatical correctness and semantic appropriateness driving the model prediction.
Recent work addresses this issue by proposing a *contrastive* formulation of such methods, producing *counterfactual explanations* for why the model predicts token $w$ instead of an alternative token
$o$.
- **Training data attribution**: Another dimension of input attribution involves the identification of *influential training examples* driving specific model predictions at inference time.
These approaches are commonly referred to as *training data attribution* (*TDA*) or *instance attribution* methods and were applied to identify data *artifacts* and sources of *biases* in language models' predictions.

#### Model Component Attribution

Early studies on the importance of transformers LMs components highlighted a high degree of *sparsity* in model capabilities.
These results motivated a new line of research studying how various components in an LM contribute to its wide array of capabilities.

- **Logit Attribution**: The *Direct Logit Attribution* (*DLA*) for a component $c$ expresses the contribution of $c$ to the *logit* of the *predicted* token, using the linearity of the model's components.
Can be applied to *FFN* blocks and *attention* heads.
The *Logit Difference* (*LD*) is the difference in logits between two tokens ($w$ and $o$).
*DLA* can be extended to measure direct *logit difference* attribution (*DLDA*).
Similarly to the *Contrastive Attribution* framework, a positive DLDA value suggests that component $c$ promotes token $w$ more than token $o$.
- **Causal Interventions**: We can view the computations of a transformer-based LM as a causal model, and use causality tools to shed light on the *contribution* to the prediction of each model *component* $c \in C$ across different *positions*.
The causal model can be seen as a *directed acyclic graph* (*DAG*), where *nodes* are model computations and *edges* are activations.
We can *intervene* in the model by changing some node's value $f_c(x)$ computed by a model component in the *forward pass* on target input $x$, to those from another value $\tilde h$, which is referred to as *activation patching*.
We can express this intervention using the **do-operator** as $f(x|do(d^c(x) = \tilde h))$.
We then measure *how much the prediction changes* after patching using a *diff* function (choices are *KL divergence*, *logit/probability difference*).
A common approach is to create a *counterfactual* dataset with distribution $P_{patch}$, where some input signals regarding the task are *inverted* (alternatively, the same interventions can be performed in a *denoising* setup, where the patch is taken from the *clean/target* run and applied over the *patched* run on *source/corrupted* input).
This approach leads to two (four) distinct types of *ablation*:
    - **Resample intervention**, where the patched activation is obtained from a *single* example of $P$ patch.
    - **Mean intervention**, where the *average* of activations of multiple $P$ patch examples is used for patching.
    - **Zero intervention**, where the activation is substituted by a *null* vector.
    - **Noise intervention**, where the new activation is obtained by running the model on a *perturbed input*.
- **Circuits Analysis**: The *Mechanistic Interpretability* (*MI*) subfield focuses on *reverse-engineerin*g neural networks into *human-understandable* algorithms.
Recent studies in MI aim to uncover the existence of *circuits*, which are a subset of *model components* (subgraphs) interacting together to solve a task.
*Activation patching* propagates the effect of the intervention throughout the network by *recomputing* the activations of components after the *patched locatio*n.
The changes in the model output allow estimating the *total effec*t of the model component on the prediction.
*Path patching* generalizes the *edge patching* approach to multiple *edges*, allowing for a more *fine-grained* analysis.
In general, we can apply *path patching* to any path in the network and measure composition between *attention heads*, *FFNs*, or the effects of these components on the *logits*.
*Circuit analysis* based on *causal intervention methods* presents several *shortcomings*: it demands significant *efforts* for designing the *input templates* for the task to evaluate, isolating *important subgraphs* after obtaining component importance estimates requires *human inspection* and *domain knowledge* and it has been shown that interventions can produce s*econd-order effects* in the behavior of downstream components (in some settings even eliciting compensatory behavior akin to *self-repair*).
A handful of methods exist to overcome these *limitations*.
Another line of research deals with finding *interpretable high-level causal abstractions* in lower-level neural networks.
These methods involve a *computationally expensive search* and assume high-level variables align with groups of units or neurons.

### Information Decoding

Fully understanding a model *prediction* entails localizing the *relevant* parts of the model, but also comprehending what information is being extracted and processed by each of these components.
While there is no universally agreed-upon definition of a *feature*, it is typically described as a *human-interpretable property* of the input, which can be also referred to as a *concept*.

#### Probing

**Probes** serve as tools to analyze the *internal representation*s of neural networks.
Generally, they take the form of *supervised* models trained to predict input properties from the *representations*, aiming to asses *how much* information about the property is encoded in them.
Although performance on the *probing* task is interpreted as evidence for the amount of information encoded in the representations, there exists a *tension* between the ability of the *probe* to *evaluate* the information encoded and the probe *learning* the task itself.
Several works propose using *baselines* to *contextualize* the performance of a probe.

Probing techniques have been largely applied to analyze Transformers in NLP.
Additionally, for *encoder models*, some studies have analyzed where *syntactic* information is stored across the *residual stream* suggesting a *hierarchical* encoding of language information, with *part-of-speech*, *constituents*, and *dependencies* being represented earlier in the network than *semantic roles* and *coreferents*, matching traditional handcrafted NLP pipelines.
Importantly, highly accurate probes indicate a *correlation* between input representations and labels, but do not provide evidence that the model is using the encoded information for its predictions.

#### Linear Representation Hypotesis

The **linear representation hypothesis** states that *features* are encoded as *linear subspaces* of the *representation space*.
For example, Word2Vec word embeddings are shown to capture *linear* syntactic/semantic word relationships.
Instances of *interpretable neurons* (neurons that fire *consistently* for specific input features, either *monosemantic* or *polysemantic*), also exemplify features represented as *directions* in the neuron space. Recent work suggests the linearity of concepts in representation space is largely driven by the *next-word-prediction training objective*
and *inductive biases* in *gradient descent* optimization.

Feature *directions* can be found in LMs using *linear classifiers* (*linear probes*).
These models learn a *hyperplane* that separates representations associated with a *particular feature* from the *rest*.
The normal vector to that *hyperplane* can be considered the direction representing the *underlying* feature.
This linear feature representation was exploited to *erase concepts*, preventing linear classifiers from detecting them in the representation space.

A fundamental problem of *probing* lies in its *correlational*, rather than *causal*, nature.
Other unsupervised methods for computing features directions include *Principal Component Analysis*, *K-Means*, or *difference-in-means*.

A *representation* produced by a model layer is a *vector* that lies in a $d$-dimensional space.
Neurons are the special *subset* of representation units right after
an element-wise *non-linearity*.
Although previous work has identified neurons in models corresponding to *interpretable features*, in most cases they respond to apparently unrelated (*polysemantic*) features.
Two main reasons can explain *polysemanticity*.
Firstly, features can be represented as *linear combinations* of the *standard basis vectors* of the neuron space, not corresponding to the basis elements themselves.
Therefore, each feature is represented across many individual neurons, which is known as *distributed representations*.
Given the extensive *capabilities* and *long-tail knowledge* demonstrated by LLMs, it has been hypothesized that models could encode **more** features than they have dimensions, a phenomenon called **superposition**.
It has been shown on toy models trained on *synthetic* datasets that *superposition* happens when *forcing sparsity* on features and, more recently, evidence of *superposition* in the early layers of a transformer language model was shown using *sparse* linear probes.

#### Sparse Autoencoders

A possible strategy to *disentangle* features in *superposition* involves finding an *overcomplete feature basis* via *dictionary* learning.
Autoencoders with *sparsity regularization*, also known as *sparse autoencoders* (*SAEs*), can be used for dictionary learning by optimizing them to reconstruct internal representations of a neural network exhibiting superposition while simultaneously promoting feature *sparsity*.
Since the output weights of each *SAE* feature interact *linearly* with the *residual stream*, we can measure their direct *effect* on the *logits* and their composition with later layers' components.
The goal of *SAEs* is to learn *sparse* reconstructions of representations.
To assess the quality of a trained *SAE* in achieving this it is common to compute the *Pareto frontier* of *two* metrics on an *evaluation set*.
These metrics are:
- **The L0 norm** of the feature activations vector, which measures how many features are "*alive*" given an input.
- **The loss recovered**, which reflects the percentage of the original *cross-entropy loss* of the LM across a *dataset* when substituting the original representations with the *SAE reconstructions*.

A summary statistic is the **feature density histogram**.
*Feature density* is the proportion of tokens in a dataset where a *SAE feature* has a *non-zero* value.
By looking at the *distribution* of *feature densities* we can distinguish if the *SAE* learnt features that are *too dense* (activate too often) or *too sparse* (activate too rarely).
Finally, the degree of *interpretability* of *sparse* features can be estimated based on their *direct logit attribution* and *maximally* activating examples.

The *sparsity penalty* used in *SAE* training promotes *smaller* feature activations, *biasing* the reconstruction process towards *smaller norms* (phenomenon is known as *shrinkage*).
This issue was addressed by proposing *Gated Sparse Autoencoders* (*GSAEs*) and a complementary *loss function*.
*GSAE* is inspired by *Gated Linear Units*, which employ a *gated ReLU* encoder to decouple feature *magnitude estimation* from feature *detection*.

#### Decoding in Vocabulary Space

A sensible way to approach decoding the information within models' representations is via *vocabulary tokens*.

The **logit lens** proposes *projecting intermediate residual stream states*.
The *logit lens* can also be interpreted as the *prediction* the model would do if *skipping* all later layers, and can be used to analyze how the model *refines* the prediction throughout the forward pass.
However, the *logit lens* can fail to elicit *plausible* predictions in some particular models, and this phenomenon have inspired researchers to train **translators** (using *linear mappings* or a *affine transformations*), which are functions applied to the *intermediate representations* (or *attention heads* in the case of the *attention lens*) prior to the *unembedding projection*.

**Patchscopes** is a framework that *generalizes patching* to decode information from *intermediate representations*.
*Patching* an activation into a *forward pass* serves to evaluate the output change with respect to the original *clean run*.

As seen in previous sections, $W_{OV}^h$ , $W_{out}$ , $W_{in}$ and $W_{QK}$ interact *linearly* with the *residual streams*. 
*Matrix weights* can be analyzed in vocabulary space by projecting them by $W_U$, and find that some *weight matrices* interact with tokens with *related* semantic meanings.
These matrices can be factorized via the **Singular Value Decomposition** (**SVD**). In the "*thin*" *SVD*, the *largest right singular vectors* represent the *directions* along which a linear transformation *stretches* the most.
By *projecting* the *top right singular vectors* onto the *vocabulary space* via the *unembedding matrix* $W_U$ we reveal the tokens the matrix *primarily* interacts with.
The *projection* of weight matrices can be extended to the *backward pass*.
Specifically, the **backward lens** projects the *gradient matrices* of the *FFNs* to study how new information is stored in their weights.

An extension of the *logit lens*, the **logit spectroscopy**, allows a *fine-grained* decoding of the information of internal representations via the *unembedding matrix* $W_U$.
*Logit spectroscopy* considers splitting the *right singular matrix* of $W_U$ into $N$ bands.
If we consider the *concatenation* of matrices associated with different *bands*, we form a matrix whose rows span a *linear subspace* of the vocabulary space.

The features encoded in model neurons or representation units have been largely studied by considering the inputs that *maximally activate* them.
Selecting examples from an existing dataset has been used in language models to *explain* the features that units and neurons respond to.
However, just relying on *maximum activating* dataset examples can result in "*interpretability illusions*", as different *activation ranges* may lead to *varying* interpretations.
*Maximally activating inputs* can produce *out-of-distribution* behaviors, and were recently employed to craft *jailbreak* attacks aimed at eliciting unacceptable model predictions.

Modern LMs can be prompted to provide *plausible sounding justifications* for their own or other LMs' *predictions*.
This can be seen as an *edge case* of *information decoding* in which the predictor itself is used as a *zero-shot explainer*.

## Discoveries

### Attention Block

Each *attention head* consists of a *QK circuit* and an *OV circuit*.
The *QK circuit* computes the *attention weights*, determining the *positions* that need to be *attended*, while the *OV circuit* *moves* (and *transforms*) the *information* from the attended position into the current *residual stream*.
Our understanding of the *specific features* encoded in the subspaces employed by circuit operations is still limited.

#### Attention Heads with Interpretable Attention Weights Patterns

- **Positional heads**: some heads attend mostly to *specific positions* relative to the token processed.
Specifically, attention heads that attend to the *token itself*, to the *previous token*, or to the *next position*.
- **Subword joiner heads**: these heads attend exclusively to *previous tokens* that are *subwords* belonging to the *same word* as the currently processed token.
- **Syntactic heads**: some attention heads attend to tokens having *syntactic roles* with respect to the *processed token* significantly more than a random baseline.
Particularly, certain heads specialize in given *dependency relation* types.
- **Duplicate token heads**: duplicate token heads attend to *previous occurrences* of the *same token* in the *context* of the *current token*.

#### Attention Heads with Interpretable QK/OV Circuits

- **Copying heads**: several attention heads in transformer LMs have *OV matrices* that exhibit *copying* behavior.
- **Induction heads**: an *induction* mechanism that allows language models to *complete patterns*.
This mechanism involves two heads in different layers composing together, specifically a *Previous Token Head* (*PTH*) and an *induction head*.
The *induction* mechanism learns to increase the *likelihood* of token $B$ given the sequence $A B ... A$, irrespective of what $A$ and $B$ are.
To do so, a *PTH* in an early layer *copies* information from the *first instance* of token $A$ to the *residual stream* of $B$, specifically by writing in the subspace the *QK circuit* of the induction head reads from (*K-composition*).
This makes the *induction head* at the *last position* to attend to token $B$, and subsequently, its copying *OV circuit* increases the logit score of $B$.
- **Copy suppression heads**: *reduce* the *logit score* of the token they *attend to*, only if it appears in the *context* and the current *residual stream* is *confidently predicting* it.
This mechanism was shown to improve overall *model calibration* by avoiding *naive copying* in many contexts.
The *OV circuit* of a *copy suppression head* can *copy-suppress* almost all of the tokens in the model's vocabulary when attended to.
*Copy suppression* is also linked to the *self-repair* mechanism since *ablating* an *essential* component deactivates the *suppression behavior*, compensating for the *ablation*.
- **Successor heads**: given an input token belonging to an element in an *ordinal sequence*, the '*effective OV circuit*' of the *successor heads* increases the logits of tokens corresponding to the next elements in the sequence.

#### Other Attention Properties

- Some **specialized domain heads** exist, which contribute only within *specific input domains*, such as non-English contexts, coding sequences, or specific topics.
- A head may attend to *special tokens* when its *specialized* function is *not applicable* (*no-op* hypothesis).
In auto-regressive LMs, these patterns are observed mainly in the *Beginning Of Sentence* (*BOS*) token, although other tokens play the same role.
Allowing attention mass on the *BOS token* is necessary for *streaming generation*, and performance degrades when the BOS is omitted.
Early FFNs in Llama 2 write *relevant information* (for the **attention sink mechanism** to occur in *later layers*) into the *residual stream* of BOS.
These FFNs write into the *linear subspace* spanned by the *right singular vectors* with the *lowest* associated singular values of the *unembedding matrix*.
This is referred as a *dark subspace* due to its *low interference* with *next token prediction*, and finds a significant correlation between the average *attention* received by a token and the existence of these *dark signals* in its *residual stream*.
These *dark signals* reveal as massive activation values acting as *fixed biases*, a crucial prerequisite for the *attention sink* mechanism to take place. 

### Feedforward Block

The behavior of neurons in language models has been extensively studied, with examinations focusing on either their *input* or *output behavior*.

In the context of **input behavior analysis**, neurons firing exclusively on *specific position ranges* were discovered.
Other discoveries include *skills neurons*, whose activations are correlated with the task of the input prompt, *concept-specific neurons* whose response can be used to predict the presence of a concept in the provided context.
Neurons responding to other *linguistic* and *grammatical features* have also been found.

Regarding the **output behavior of neurons** the *key-value* memory perspective of FFNs offers a way to understand neuron's weights.
Specifically, using the *direct logit attribution* method we can measure the neuron's effect on the logits.
Some neurons promote the prediction of tokens associated with particular *semantic* and *syntactic concepts*.
A small set of neurons in later layers is responsible for making *linguistically acceptable predictions*.
Some neurons interact with *directions* in the *residual stream* that are similar to the *space and time* feature directions extracted from probes.
Neurons *suppressing improbable continuations* have recently been identified.

Recent work highlighted the presence of **polysemantic neurons** within language models.
Notably, most early layer neurons specialize in sets of *n-grams*, functioning as *n-gram detectors*, with the majority of neurons firing on a large number of n-grams.
Models internally perform "*de-/re-tokenization*", where neurons in *early layers* respond to *multi-token* words or *compound* words, mapping tokens to a more *semantically meaningful* representation (*detokenization*).
In contrast, in the *latest layers*, neurons aggregate *contextual representations* back into single tokens (*re-tokenization*) to produce the next-token prediction.

Whether *different models learn similar features* remains an open question.
Across the cluster of **universal neurons** that *shared activations* on the *same inputs* between models there is a higher degree of *monosemanticity*.
This group includes *alphabet neurons*, which activate in response to tokens representing *individual letters* and on tokens that start with the letter, supporting the *re-tokenization* hypothesis.
Additionally, there are *previous token neurons* that fire based on the preceding token, as well as unigram, position, semantic, and syntax neurons.
In terms of *output behavior*, *universal neurons* include *attention (de)activation neurons*, responsible for controlling the amount of *attention* given to the *BOS token* by a subsequent *attention head*.
Some neurons may even act as *entropy neurons*, modeling the model's *uncertainty* over the next token's prediction.

### Residual Stream

We can think of the **residual stream** as the main *communication channel* in a transformer.
The "*direct path*" connecting the input *embedding* with the *unembedding* matrix does **not** move information between positions, and mainly models *bigram statistics*, while the latest biases in the network, localized in the prediction head, are shown to *shift predictions* according to *word frequency*, promoting *high-frequency* tokens.
*Alternative paths* involve the *interaction* between components, potentially doing more *complex computations*.
The *norm* of the residual stream *grows exponentially* along the layers over the forward pass of multiple transformer LMs.
A similar *growth rate* appear in the norm of the output matrices writing into the *residual stream*, unlike input matrices, which maintain *constant norms* along the layers.
It is hypothesized that some components perform *memory management* to *remove information* stored in the *residual stream*.

**Outlier dimensions** have been identified within the *residual stream*.
These *rogue dimensions* exhibit *large magnitudes* relative to others and are associated with the generation of *anisotropic representations*. 
*Anisotropy* means that the residual stream states of random pairs of tokens tend to point towards the *same direction*.
Furthermore, *ablating* outlier dimensions has been shown to significantly decrease downstream performance, suggesting they encode *task-specific* knowledge.
The presence of *rogue dimensions* has been hypothesized to stem from *optimizer choices*.

The *specific features* encoded within the *residual stream* at various layers remain uncertain, yet **sparse autoencoders** offer a promising avenue for improving our understanding.
Recently, *SAEs* have been trained to reconstruct *residual stream states* in small language models showing highly *interpretable* features.
Since *residual stream* states gather information about the *sum* of previous components' outputs, inspecting SAE's features can illuminate the process by which they are *added* or *transformed* during the *forward pass*.
Given the type of features intermediate FFNs and attention heads interact with, we also expect the *residual stream* at *middle layers* to encode highly *abstract* features.
Based on output behavior via the *logit lens*, *local context* features promote *small* sets of tokens and highlight the presence of *partition features*, which *promote* and *suppress* two distinct sets of tokens.
Recent studies have shown that language models create vectors representing functions or tasks given *in-context* examples, which are found in *intermediate layers*.

### Emergent Multi-component Behaviors

In order to explain the remarkable performance of transformers, we also need to account for the **interactions between the different components**.

Recent evidence suggests that *multiple attention heads* work *together* to create "*function*" or "*task*" vectors describing the task when given in-context examples.
Intervening in the residual stream with those vectors can produce outputs in accordance with the encoded task on novel *zero-shot* prompts.
Specifically, *middle layers* process the request, followed by a *retrieval step* of the entity from the *context* done by *attention heads* at *later layers*.

Individual neurons within *downstream* FFNs activate according to the output of *previous attention heads*, interacting in specific contexts.
In the *Indirect Object Identification* (*IOI*) task the model is given inputs of the type "*When Mary and John went to the store, John gave a drink to ___*".
The initial clause introduces two names (Mary and John), followed by a secondary clause where the two people exchange an item.
The correct prediction is the name not appearing in the second clause,referred to as the Indirect Object (Mary).
A circuit found in GPT-2 Small mainly includes:
- **Duplicity signaling**: *duplicate token heads* at position S2, and an *induction mechanism* involving previous token heads at S1+1 signal the *duplicity* of S.
This information is read by *S-Inhibition heads* at the last position, which write in the *residual stream* a token signal, indicating that S is repeated, and a position signal of the S1 token.
- **Name copying**: *name mover heads* in *later layers* copy information from names they attend to in the context to the last *residual stream*.
However, the signals of the previous layers *S-Inhibition heads* modify the query of name mover heads so that the *duplicated name* (in S1 and S2) is *less attended*, favouring the copying of the *Indirect Object* (*IO*) and therefore, pushing its prediction.

In addition, **negative mover heads**, which are instances of *copy suppression heads* *downweight* the probability of the *IO*.
While the IOI is an *attentioncentric circuit*, examples of circuits involving both *FFNs* and *attention heads* are also present.

The functionality of the circuit components remains consistent after *fine-tuning* and benefits of *fine-tuning* are largely derived from an improved ability of circuit components to encode important *task-relevant* information rather than an overall *functional rearrangement*.
*Fine-tuned activations* are also found to be *compatible* with the base model despite no explicit tuning constraints, suggesting the process produces *minimal changes* in the overall *representation space*.
Additionally, *circuits* are *not exclusive*, i.e. the same model components might be part of several circuits.

Transformer models were observed to converge to *different algorithmic solutions* for tasks at hand.
Convincing evidence exists on the relation between *circuit emergence* and *grokking*, i.e. the sudden emergence of near-perfect generalization capabilities for simple symbol manipulation tasks at late stages of model training.
The *grokking phase transition* can be seen as the emergence of a *sparse circuit* with *generalization capabilities*, replacing a *dense subnetwork* with *low generalization capacity*.
This might happen because dense memorizing circuits are *inefficient* for compressing large datasets.

#### Hallucinations

The generation of factually incorrect or nonsensical outputs is considered a significant *limitation* in the practical usage of language models.
While some techniques for detecting hallucinated content rely on *quantifying the uncertainty of model predictions*, most alternative approaches engage with *model internal representations*.
Approaches for detecting hallucinations directly from the representations include *training probes* and *analyzing the properties of the representations leading to hallucinations*.

A related area of research with overlapping goals is that of hallucination detection in machine translation (MT).
An MT model is considered to *hallucinate* if its output contains *partially* or *fully* detached content from the *source sentence*.
*Prediction probabilities* of the generated sequence and *attention distributions* have been used to detect *potential errors* and model hallucinations.
Recently, methods measuring the *amount of contribution* from the *source sentence tokens* were found to perform on par with external methods based on *semantic similarity* across several categories of model
hallucinations.
*Detection methods* show complementary performance across *hallucination categories*, and simple *aggregation strategies* for internals-based detectors outperform methods relying on external semantic similarity or quality estimation modules.

The underlying mechanisms involved in the prediction of hallucinated content for LLMs remain largely unexplained.

Recent research has delved into the *internal mechanisms* through which language models **recall factual information**, which is directly related to the *hallucination* problem in LLMs.
A common methodology involves *studying tuples* $(s, r, a)$, where $s$ is a *subject*, $r$ a *relation*, and $a$ an *attribute*.
The model is prompted to predict the *attribute* given the *subject* and *relation*. 
*Causal interventions* can be used to *localize* a *mechanism* responsible for *recalling factual knowledge* within the language model.
*Early-middle FFNs* located in the last subject token add information about the subject into its *residual stream*.
On the other hand, information from the *relation* passes into the last token residual stream via *early attention heads*.
Finally, *later layers* attention heads extract the right attribute from the last subject *residual stream*.

Subsequent research has moved from localizing model behavior to *studying the computations performed to solve this task*.
Attributes of entities can be *linearly decoded* from the *enriched subject residual stream*.
More precisely, using the *direct logit attribution* by each token via the *attention head* it is possible to identify **subject heads** responsible for extracting attributes from the *subject* independently from the *relation* (not attending to it), as well as **relation heads** that promote *attributes* without being causally dependent on the *subject*.
Additionally, a group of **mixed heads** generally favor the *correct attribute* and depend on both the *subject* and *relation*.
The combination of the different heads' outputs, each proposing different sets of *attributes*, together with the action of some downstream FFNs resolve the *correct prediction*.

Recent works aim to shed light on how the model engages in **factual recall vs. grounding**.
Following the aforementioned *(subject, relation, attribute)* structure of facts, an answer is considered to be *grounded* if the *attribute* is *consistent* with the *information* in the *context of the prompt*.
Two main type of attention heads emerge: **in-context heads** and **memory heads**, which respectively favor the *in-context answer* and the *memorized answer*, showing a "*competition*" between mechanisms.
*FFNs* in the last token of the subject have *higher contributions* on *ungrounded* (*memorized*) answers as opposed to *grounded* answers, while suggesting that grounding could be a more *distributed process* lacking a specific localization.
The recall of "*memorized*" idioms largely depends on the *updates of the FFNs in early layers*, providing further evidence of their role as
a storage of memorized information.
This is further observed in the study of *memorized paragraphs*, with lower layers exhibiting larger *gradient flow*.

*Factual information* encoded in LMs might be incorrect *from the start*, or become *obsolete* over time.
Moreover, *inconsistencies* have been observed when recalling factual knowledge in *multilingual* and *cross-lingual* settings, or when factual associations are elicited using *less common formulation*s.
This sparked the interest in developing *model editing* approaches able to perform **targeted updates** on model factual associations with *minimal impact on other capabilities*.
While early approaches proposed edits based on *external modules* trained for *knowledge editing*, recent methods employ *causal interventions* to *localize knowledge neurons* and *FFNs* in one or more layers.
However, model editing approaches still present several *challenges* including the risks of *catastrophic forgetting* and *downstream performance loss*.