# Nicolò 29/5/24

- Nicolò is working on mapping the activation circuit:
    - Check if there exists a summation algorithm inside layers
    - Generate arithmetic operation dataset
    - Trace FFNN/attention activations
    - Look at highest activations
    - Test with ablations

- Reiterate analysis of relevance geometric properties of embeddings for transformers
    - Interest is not enough to justify the work, there need to be clear research interests
    - Prioritize literature
    - Check linearity of transformer transformations
    - Possible justification of rogue dimensions

# Meeting 30/5/24

- Positional information
    - In recent models there is no positional information in the residuals
    - The only positional information is injected in the query-key computation

- Paper reviews
    - Main takeaways
        - Reviewers are playing it safe, they want a paper that is 100% clear and polished
    - Ideas going forward
        - Add statistical significance to experiments
        - Experiment on more datasets/benchmarks
        - Bulk up literature review
        - Try running on latest models
    - Invert order of Q1 and Q2
    - Nicolò
        - Layer abstraction -> Specificity
        - POS tags ~ flash out it more
            - More preprocessing with NER
            - Look at how similar the attention head was for each attention head to a semantic/syntactic parse tree
            - Parallel with minimum spanning tree over the sequence of tokens and find where the minimum spanning tree encodes the noun phrase
            - Take a bunch of multi-token NEs, encode their embedding as the mean and feed it to the model OR search it inside the computation of the model