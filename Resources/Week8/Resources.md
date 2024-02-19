# Previous Meeting Ideas

## 08/02/24 Group Meeting

### Ideas

- Possibility of using [Capybara model](https://huggingface.co/NousResearch/Nous-Capybara-34B) as an alternative to Llama.
- Attention weights have no summation to 1 requirement for all heads.
- Softmax aggregates all attention heads.
- Heads could actually perform similar tasks between one another, and there could be multiple only to reduce variance between results.

### Tasks

- Is it possible to calculate the contribution of each single attention head, are all heads weighted equally?
    - Possibilities:
        - Calculate entropy distribution of heads.
        - Apply weights to residual.
        - Get L2 norm of vector that is aggregation of total heads before non-linear activation function present in linear layer.

- Is there correlation between attention weights and the change in meaning of hidden representations from the input token to the output token?

- Make box color represent the certainty of represented token in the visualization.

- Use simple examples.

- What is the right representation of tokens at any point inside the transformer?
    - Possibility to extrapolate a combination of concepts:
        - Find most likely token and subtract it -> Find most likely token and subtract it -> ...
        - Try to propagate token to the end (what is the most likely sequence of tokens at that point?).

- Identify where the carry happens.

- Test with arbitrary information but that the model knows (therefore it is not stored in the embedding).


## Nicolò 13/2

## Tasks

### Visualization

- Create a dataclass to store standardized visualization data for future reference and generality.

- Instead of plotting lines representing attentions, change the color of the connected blocks (while only leaving lines over a certain threshold).

- Put hidden representation token inside visualization blocks instead of under them.

- Put more distance between visualization blocks.

- Insert points between layer to better display aggregated attentions.
    - Image 1 in [Notes 1](13-02-24_Notes_1.png").

### Attention Heads Contribution

- Give priority to "L2 norm" investigation

- Understanding Multi-head attention:
    - Every head performs attention computation on the input and produces a resulting embedding that gets compressed by the $W_d$ matrix.
    - All resulting compressed embeddings get concatenated into a new vector.
    - The concatenated vector gets compressed by the $W_o$ matrix.
    - Image 2 in [Notes 1](13-02-24_Notes_1.png") and Image in [Notes 2](13-02-24_Notes_2.jpg")
    - Look at attention implementations of LLama2 and GPT2.

- Main questions: 
    - How and how much does each head contribute to the final embedding?
    - What is selecting each head?

- Ideas:
    - Cosine distance is a good measure to compare single-head embeddings and final embedding after multiplication with $W_o$.
    - Calculate $\tilde{E}$ by only considering contributes of a single head and then compute to compare them $1 - |d(E_{out}, \tilde{E}_{out})|$ (or alternatively look at which $\tilde{E}_{out}$ has higher norm).

- Technically $\sum{\tilde{E}_{out}} = E_{out}$

- Implementation-wise slicing matrices is better than putting 0s in them.