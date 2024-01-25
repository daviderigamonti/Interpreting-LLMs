# Previous Meeting Ideas

## 12/12/23 w/ Nicolò, Vincenzo

- Embedding Arithmetic
    - Try embedding experiment using cosine similarity
    - Look for specialized dataset for performing embedding arithmetic and similar
        - Gensim, gensim.test.utils
        - https://radimrehurek.com/gensim/models/keyedvectors.html

- Embedding Arithmetic additional ideas
    - Try with output embedding instead of input embedding
        - Difficult with autoregressive models
        - Maybe LaMa does sentence embedding during pretraining
    - Understand how the embedding of a token and its relationships change throughout layers of a model

## 13/12/23 w/ Nicolò, Vincenzo

- Embedding Arithmetic Notebook
    - Try using Pearson/Spearman Correlation to evaluate word similarity results
    - Visualize top5, top10, or other kinds of actual metrics instead of the rank score

- LangChain
    - High level library to wrap models and perform practical tasks

- New Ideas
    - Are embedding spatial properties preserved through the network layers? If so, for an autoregressive causal model there must a point where the model starts predicting the next token and the representation changes, is it gradual or instantaneous?
        - Experiment 1 - Repeat arithmetic tests between model layer embeddings
            - A: Select words that can be represented by a single token (for both inputs and solution)
            - B: Use average on the same run to represent multi-token words
            - C: Use average on the independent run to represent multi-token words
            - D: Weighted average using attention weights
        - Experiment 2 - If experiment 1 doesn't work, is the problem in the model structure (autoregressive) or in the non-linear transformations?
            - A: Try with an encoder model