# Papers

# Previous Meeting Ideas

## 25/01/24 Group Meeting

### Ideas

- Output embeddings can be different from input embeddings, and this could affect the internal representation of the model and the representation inconsistent between layers.
- Cosine Distance and Euclidean Distance tend to be very similar inside the unit circle.
- Layer Normalization performs normalization through the whole input vector shifting its range inside the unit range; its parameters do NOT correspond to the training mean/standard deviation, but are the contribution of the normalization formula and a bias.
- Nicolò's and Davide's research topics may actually be similar ("What happens inside a model when it fails?" vs "How does the memory of an LLM work?").
- Self-attention layers acts as aggregation for information and FC layers work as lookup, however self-attention also includes a projection operation, which is able to also add information.

### Tasks

- Share meeting notes on discord channels.
- Check if GPT uses same embeddings at its input and output.
- Try performing research by giving a model tasks rather than verifying the properties of its components:
    - Can be multiple simple tests.
    - E.g.: Visualize the inter-layer contributions for a summation (1230912 + 12391989 = ?) and figure out how mathematical carries are represented.
- Find out when the model starts predicting the next word by:
    - Taking a common sequence/sentence with an obvious token that should have high output probability.
    - Trace the token through the model prediction and find where it starts getting predicted.
    - Task should be easier if input and output embeddings are equal.