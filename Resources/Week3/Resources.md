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
        - ROME is compared with Fine-Tuning (FT), Constrained Fine-Tuning (FT+L) and hypernetworks (KE, MEND) on a Zero-Shot Relation Extraction (zsRE) task measuring efficacy, paraphrase and specificity and on a custom CounterFact dataset to evaluate counterfactual edits measuring efficacy (Efficacy Score and Magnitude), generalization (Paraphrase Score and Magnitude), specificity (Neighborhood Score and Magnitude), consistency (RS), fluency (GE) and the harmonic mean between ES,PS and NS as Score (S). 
        - ROME is competitive against other methods on zsRE (even though it falls short against zsRE custom-tuned hypernetworks) and on CounterFact all methods besides ROME present one or both of the following problems while ROME demonstrates bothe generalization and specificity:
            - F1: overfitting on counterfactual statement -> no generalization
            - F2: underfitting and ignoring counterfactual statement
        -  Midlayer MLP modules can store factual associations that generalize beyond specific surface forms, while remaining specific to the subject.


# Benchmarks


## Question Answering
> Ability of model to remember facts it was trained with, e.g. in what year was Barack Obama born? How tall is a male giraffe?

## Mathematical Reasoning
> Ability of model to perform basic mathematical reasoning: if I have four bunches of bananas containing an average of 5.5 bananas each, how many bananas do I have?

## Logical Reasoning
> Ability of model to reason logically: e.g. my dad has two kids, one of them is called John. If my brother's name is Steven, what is my name?

4) ability of the model to be polite/courteous, etc.

5) ability of the model to be creative

6) ability of the model to empathise/understand user emotion