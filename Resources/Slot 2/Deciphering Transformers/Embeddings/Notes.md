# Meeting 31/5/24

- Ideas
    - Compare results to Word2Vec as a baseline
    - Point out the fact that there is no way to transform a transformer in Word2Vec, therefore, we are checking known properties of embeddings, but the fact that we are applying them to transformers is releveant and makes it novel
    - New embedding experiment
        - Create new embedding as "paris - france + italy" or similar
        - Ask "<embedding> is the capital of" to the model

# Meeting 13/6/24

- Word2Vec BERT tokenization is a problem
        - try using nltk tokenizer?
        - Vincenzo says to look at his wrapper libraries, where he extended part of the tokenizer
- 2 possible ways to look at Llama3 underperforming
    - How do we change the math of the computation to make it work for Llama3?
    - Or maybe it's just the case that the model uses the space in a non-linear way and adds other information that is not strictly relevant to token semantics

- SVD on co-occurrences matrices performs better than Word2VEc/GLoVE

- when taking averages in space we might need to normalize before averaging

- call "average relationship/average translation" instead of delta anlogies