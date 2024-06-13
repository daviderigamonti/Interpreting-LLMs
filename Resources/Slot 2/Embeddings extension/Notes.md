# Meeting 13/6/24

- Experiment is cool, but THERE MUST BE a solid reason to the experiments we perform
    - Mark not conviced by the original reason of the experiment (trying to understand if current LLMs still exploit embeddings spatial properties for their computations)
    - Possible reasons for the experiment that need to be elaborated
        - Understand if the model ability to leverage the geometric properties does not depend on the embedding
            - we may not be testing this in the right way (arithmetic of embeddings)
        - Maybe understand if embeddings also contain disjunction or conjunction information (e.g. embedding containing both 6 and 7 until one of them is disabiguated inside the transformer computation)
        - Understand what happens when there is a token with two different meanings
        - Demonstrate that embeddings present linear properties only around boundaries of tokens

- Repetition issue
    - Repetition attention head hypothesis
        - Look at head distribution on every layer
        - Look at rows of W_o matrix, which acts as weight for each attention head
    - Ambiguous embedding hypothesis (https://arxiv.org/pdf/2210.14140)
        - Look at the embedding distance w.r.t. others and density of tokens in the space

- Issue of adding a token to the vocabulary
    - Extende vocabulary of the model with a randomly initialized embedding + arithmetic result
    - Fine-tune the model by performing incremental updates on the embedding to let it adjust

- Additional ideas to play around with
    - Use "X = Rome + Paris" and se how the model handles 2 different meanings
    - We might consider the concept obtained by using an analogy as the "purer" concept and maybe even show it by disambiguating a piece of text (e.g. changing the meaning of father to the arithemtic result and compute "Alex's father is called Tom, Tom has 2 children, one of them is called Steven, what's the name of the other one?")
        - We might even want to define a token as the average of multiple different analogies that return the same object
