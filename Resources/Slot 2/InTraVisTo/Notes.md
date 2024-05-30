# Meeting 2/5/24

- IDEAS
    - Remember that the main focus is to show how the model is recalling factual information
    - Need a dataset of examples that require different types of reasoning in order to predict the next token, for example:
        - What is the next number in the sequence 1, 2, 3, .. / 1, 2, 4, 16, ...
    - Introduce a comparison with what a Markov model could produce in terms of text

# Nicolò 29/5/24

- Residual approach emerged from literature papers:
    - Residuals are not made to directly be decoded, but they work as communication channels
    - We are nonetheless interested in using this mechanic to decode layer information in some way

- Ideas for Sankey diagram:
    - Also track various attention head contributions, possibly showing the top-k heads based on their activations

- Thesis:
    - Make the visualization tool the main contribution
    - Eventually expand by making focused experiments by employing the tool

- Need to refactor code:
    - Write down small/general UML diagram for a new class organization
    - Change model interface to include Vincenzo's libraries
    - No concrete progress towards the heatmap -> table switch

# Meeting 30/5/24

- Ideas
    - Extract weights from combinations in tuned lens-style

- Anisotropy
    - Observe entropy of logits distribution
        - High entropy -> Embeddings being close/far holds less significance

- Extra: Are layers linear transformations
    - Train a model from scratch
    - Pretraining on original dataset
    - Force independence between layers