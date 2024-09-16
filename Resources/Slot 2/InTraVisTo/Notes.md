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

# Meeting 31/5/24

- When looking at the token representation of the difference between layers:
    - Look at the difference of logits:
        - We are looking at the relationship/change ("capital city of", "masculine of",...)
    - Look at the difference of probability distribution:
        - We are looking at the importance of the tokens
    - Give the possibility to switch between the two representations

- Interface changes
    - Smaller words on intermediate/attention/ffnn (not main layers)
        - Give the possibility to remove them by using a flag
    - Embedding selection popup
        - Change "Insert your embedding" to "Embedding to change"
        - Change the color of a cell when selecting it
    - Possibly change the color of nodes by the size of the euclidean norm vector

- Bugfix
    - Check interpolation level off-by-one errors for token injection

# Nicolò 11/6/24

- 5 August deadline for EMNLP demo track
- New single visualization that merges heatmap with Sankey
    - remove heatmap
    - multiple cells for each layer
    - Sankey as a highlight of cells

# Mark 14/6/24

- Interface changes
    - Add banner at the top of page
    - Tooltips
        - "Number of token" -> "Token position"
        - Remove new lines
    - Set 12 as default font size
    - Change attention from "att_in from x to y" -> "Attention x"

- Weird "residual + attention" outputs, double check them

- What happens when we give different weights to the average between numerical embeddings?
    - 1 + 1 + 4 = 
    - e[""], e[1], e["_+"], e["_1"], e["_+"], e["_"], e["4"], e["_="]
    - 1 + 0.4e[1] + 0.6e[2] + 4 = ?

# Meeting 27/6/24

- Interface changes
    - outflows should be the same color 
    - my idea: Try hiding small weights to avoid cluttering

- Sankey
    - we are kinda visualizing a linear approximation of the system since we are simulating weighted sums without nonlinearities (assuming as)
    - mutual information way for decomposing: "how much of the information is coming from one to the"
    - Sankey is the right diagram! It's exactly the flow of information that we want to show

- Play around with easy summations and find examples where we can determine internally the calculations that are happening

# Nicolò 23/7/24

- EMNLP August 4th deadlin
- Primary Tasks
    - scrollable vis
    - overlay model substitution
    - KL Div weighted on norm
    - Look at proabilities
- Secondary Tasks
    - Fix colors

# Week 29/7/24 - 4/8/24

- Paper
    - Related works & background
        - Lens-like applications
        - 3 sections
            - Interpreting hidden states
            - Information flow
            - Injecting information
    - Research questions
        - Interpreting model
        - Let the human understand the process that the model performs in order to return an output
    - Beware of comparisons! Less comparisons and reduce "brutality"

- Application
    - Interpolation problem
    - Normalization
        - transformations:
            - Normalizing to length RMS
            - Rescaling to weight
        - Non linear transformation given by n_hid weights
        - do we normalize using the weights or just rescale?
    - Attention
        - Set attention non-highlighted opacity to 0 by default
        - Change attention opacity label name (remove)
        - Alternative ways to weight attention traces
            - Show top-k attention traces (default)
            - Hide all traces with weights under x
            - Show all traces that sum up to a probability x
    - Visual
        - Add X-AI logo
        - Change line color
        - Future hypothetical layout
            - Column on the left with parameters
            - Visualization on the right
    - Injection
        - New injection idea: change contribution of attention/residual ffn/residual
    - Fixes
        - Why input probabiliteis of embedding layer are 0.0
        - Use normalization to increase weight probabilities

# Meeting 13/09/24

- Change injection to new formula defined in .pdfs
- Interface:
    - Add XAI-lab, Politecnico di Milano logos in the top banner
    - Spell out InTraVisTO 
- Interpolation:
    - Idea of dropping quadratic interpolation
    - Are all the vectos in the input the same size?
        - Test: Show norms of vectors in input and output embeddings
        - It may make sense to first normalize embedding before performing interpolation

- Possibility of letting the user remove links from the Sankey?
- Possibility of implementing the table instead of the heatmap