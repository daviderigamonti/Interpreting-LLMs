# Papers

### [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
- [Local Copy](PDFs/locating_and_editing_factual_associations_in_gpt.pdf)
- [Annotated Copy](PDFs/Annotated/locating_and_editing_factual_associations_in_gpt_annotated.pdf)
- Main research question: Where does a LLM store its facts?
- **First approach**: Trace causal effects of hidden state activations within GPT using **causal mediation analysis** to identify modules that mediate recall of a fact about a subject.
    - **Causal mediation analysis** quantifies the contribution of intermediate variables in causal graphs (the grid of hidden states affected by attention and MLPs forms a causal graph).
    - Approach
        - Observe the internal activations of causal graph $G$ during three runs: **clean**, **corrupted** and **restored**.
            - **clean**: Pass factual prompt $x$ into $G$ and collect hidden activations.
            - **corrupted**: Add gaussian noise $\epsilon \sim \mathcal{N}(0, \nu)$ to embeddings of words that correspond to the subject entity and collect hidden corrupted activations.
            - **restored**: Proceed as in the corrupted case but for some token $\hat i$ and layer $\hat l$ substitute hidden state $h_{\hat i}^{(\hat l)}$ with the corresponding clean state.
        - The **Total Effect** (TE) is given by the difference between the probability of emitting $o$ (object of the fact, the expected answer) in the **clean** and **corrupted** runs; the *Indirect Effect* (IE) of a specific state $h_i^{(l)}$ is given by the difference between the probabilities of emitting $o$ in the **restored** (restoring exactly $h_i^{(l)}$) and **corrupted** runs.
    - Results
        - Computing average TE and IE for over 1k factual statements discovers the presence of strong causal states at an early site at the last token of the subject; by furtherly isolating attention and MLP contributions, it is underlined the essential role of MLP module computation at middle layers when recalling a fact.
        - Feedforward MLPs at a range of middle layers are decisive when processing the last token of the subject name.
        - Further hypothesis that there is no special role for the particular choice or arrangement of individual layers in the middle range as any fact could be equivalently stored in any of the middle MLP layers; to this end, verify if it's possible to store arbitrary facts inside a mid-range MLP layer by modifying its weights.
- **Second approach**: Introduction of a **Rank-One Model Editing** (ROME) method to alter the parameters that determine a feedforward layer's behavior at the decisive token.
    - **ROME** makes it possible to modify factual associations by inserting a new knowledge tuple in place of a current tuple with both generalization and specificity.
    - Approach
        - Existence of a closed form equation to modify weight matrix in order to substitute knowledge tuples by editing the corresponding key-value vectors to both select the subject and recall the fact.
        - The new key value pair needs to be computed and can be identified by feeding the model with multiple samples encoding the target subject (for the key) and relation+object (for the value) and observing the hidden states.
    - Results
        - ROME is compared with Fine-Tuning (FT), Constrained Fine-Tuning (FT+L) and hypernetworks (KE, MEND) on a Zero-Shot Relation Extraction (zsRE) task measuring efficacy, paraphrase and specificity and on a custom CounterFact dataset to evaluate counterfactual edits measuring efficacy (Efficacy Score and Magnitude), generalization (Paraphrase Score and Magnitude), specificity (Neighborhood Score and Magnitude), consistency (RS), fluency (GE) and the harmonic mean between ES, PS and NS as Score (S). 
        - ROME is competitive against other methods on zsRE (even though it falls short against zsRE custom-tuned hypernetworks) and on CounterFact all methods besides ROME present one or both of the following problems while ROME demonstrates both generalization and specificity:
            - F1: overfitting on counterfactual statement -> no generalization
            - F2: underfitting and ignoring counterfactual statement
        -  Midlayer MLP modules can store factual associations that generalize beyond specific surface forms, while remaining specific to the subject.

### [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)
- [Local Copy](PDFs/transformer_feed_forward_layers_are_key_value_memories.pdf)
- [Annotated Copy](PDFs/Annotated/transformer_feed_forward_layers_are_key_value_memories_annotated.pdf)
- Main objective: demonstrate that feedforward layers act as key-value memories in transformer architectures.
- Similarity between feedforward layer formula $FF(x) = f(x \cdot K^\top) \cdot V$ and neural memory formula $MN(x) = softmax(x \cdot K^\top) \cdot V$, where $K$ is the keys matrix and $V$ is the values matrix, both being parameter matrices.
- Claims
    - **Each key vector $k_i \in K$ captures a particular set of patterns in the input sequence**
        - Approach
            - Retrieve training examples "most associated" with a given key $k_i^\ell$ at the $\ell$-th feedforward layer and $i$-th hidden dimension by computing $ReLU(x_j^\ell \cdot k_i^\ell)$ for every prefix $x_1,...,x_j$ for every sentence and retrieving the top $t$ prefixes at layer $\ell$ with highest score.
            - Evaluate retrieved prefixes by using human experts to evaluate and describe repetitive patterns that occur in at least 3 prefixes and classify them as shallow (recurring n-grams) or semantic (recurring topic).
            - Further experiment by removing first/last/random token from input of top-50 trigger samples and measure memory coefficient score.
        - Results
            - Experts were able to identify at least one pattern for every key with an average of 3.6 identified patters and the vast majority of retrieved prefixes were associated with at least one identified pattern.
            - Key vectors in ff layers act as pattern detectors.
            - Shallow layers identify shallow patterns, while higher layers are more prone to identify demantic patterns since removing the last token has less impact w.r.t shallow layers.
    - **Each value $v_i^\ell$ can be viewed as a distribution over the output vocabulary** 
        - Approach
            - Convert each value vector $v_i^\ell$ into a probability distribution over the vocabulary under the naive assumption that all model layers operate under the same embedding space: $p_i^\ell = softmax(v_i^\ell \cdot E)$, then find the next token of $k_i^\ell$'s top-1 trigger example and find where it ranks in the value vector's distribution $p_i^\ell$.
            - Compute agreeent rate as the fraction of memory cells where the value's top prediction mathces the key's top trigger example.
        - Results
            - The agreement rate is close to zero in lower layers but exponentially increases in higher layers, implying that memory cells often store information on how to directly predict the output from the input.
            - The same might be also true for lower layers but the target embedding space is different from the output embedding space, thus the resulted distributions are useless.
            - Further exploration possible.
    - **Every feedforward layer combines multiple memories to produce a distribution that is qualitatively different from each of its component memories' value distributions and residual connections refine the results**
        - Approach
            - Using the validation set, memory cells with non-zero coefficients are observed and denoting as $top(h) = argmax(h \cdot E)$ the top prediction from the vocabulary distribution induced by vector $h$, the number of examples where the following formula is true is computed: $\forall i top(v_i^\ell) \ne top(y^\ell)$ (where $y^\ell$) is the final prediction of a feedforward layer.
            - Given $o^\ell = y^\ell + r^\ell$ where $r^\ell$ is the residual and $y^\ell$ is computed using layer normalization, they measure the number of time the probability distribution induced by the residual matches the final prediction, the probability mass $p$ that each layer's residual vector assigns to the model's final prediction and how often the residual's top prediction changes following its interaction with the feedforward layer (and if it overrides it).
        - Results
            - The majority of outputs is clearly a compositional result of all the memory cells in the feedforward layer (more for shallow layers and less for top layers) and never the result of a single dominant memory cell, only exception is for common few-word patterns and stopwords.
            - While analyzing residual connections it emerges that roughly a third of the model's predictions are determined in the bottom few layers, implying that the majority of "hard" decisions occur in shallow layers and the model's confidence in its predictions grows in a similar pattern.
            - In the vast majority of cases the residual's top prediction ends up being the model's prediction with the agreement of the ffn layer, rarely the ffn may correspond to the model's prediction but the residual prediction almost never agrees and the resulting output is a compromise of the two. It may be possible that the feedforward layers acts as an elimination mechanism to "veto" the top prediction in the residual.

# Benchmarks

https://deepgram.com/learn/llm-benchmarks-guide-to-evaluating-language-models

## Question Answering
> Ability of model to remember facts it was trained with, e.g. in what year was Barack Obama born? How tall is a male giraffe?

### The Stanford Question Answering Dataset (SQuAD)
- https://paperswithcode.com/dataset/squad
- Tasks: Question Answering

### Natural Questions 
- https://paperswithcode.com/dataset/natural-questions
- Tasks: Question Answering, Passage Retrieval, Open-Domain Question Answering

### TriviaQA
- https://paperswithcode.com/dataset/triviaqa
- Tasks: Question Answering, Open-Domain Question Answering

### HotpotQA
- https://paperswithcode.com/dataset/hotpotqa
- Tasks: Question Answering 

## Mathematical Reasoning
> Ability of model to perform basic mathematical reasoning: if I have four bunches of bananas containing an average of 5.5 bananas each, how many bananas do I have?

### GSM8K
    - https://paperswithcode.com/dataset/gsm8k
    - Tasks: Arithmetic Reasoning

### MATH
    - https://paperswithcode.com/dataset/math
    - Tasks: Math Word Problem Solving 

## Logical Reasoning
> Ability of model to reason logically: e.g. my dad has two kids, one of them is called John. If my brother's name is Steven, what is my name?

### ReCoRD
- https://paperswithcode.com/dataset/record
- Tasks: Common Sense Reasoning

### CommonsenseQA 
- https://paperswithcode.com/dataset/commonsenseqa
- Tasks: Common Sense Reasoning

### AI2 Reasoning Challenge (ARC)
    - https://paperswithcode.com/dataset/arc
    - Tasks: Common Sense Reasoning

## Politeness
> Ability of the model to be polite/courteous, etc.

### Typologically Diverse Politeness (TyDiP)
    - https://paperswithcode.com/dataset/tydip

## Creativity
> Ability of the model to be creative

## Empathy
> Ability of the model to empathise/understand user emotion

### IMDb Movie Reviews
- https://paperswithcode.com/dataset/imdb-movie-reviews
- Tasks: Text Classification, Sentiment Analysis

### Multi-Perspective Question Answering Opinion Corpus (MPQA)
- https://paperswithcode.com/dataset/mpqa-opinion-corpus
- Tasks: Sentiment Analysis, Fine-Grained Opinion Analysis

## Summarization

### CNN/Daily Mail 
- https://paperswithcode.com/dataset/cnn-daily-mail-1
- Tasks: Summarization

### WikiHow
- https://paperswithcode.com/dataset/wikihow
- Tasks: Summarization

### Extreme Summarization (XSum)
- https://paperswithcode.com/dataset/xsum
- Tasks: Seq2Seq Language Modeling, Summarization, Text Summarization