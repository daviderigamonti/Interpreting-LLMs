# Previous Meeting Ideas

## 07/03/24 Group Meeting

### Ideas

Head Contribution:
- Head contribution plots should have a reason to exist, maybe focusing on some specific heads.
- Singular head contribution may be too specific to have any meaningful display or patterns. Including residuals may help to paint a bigger picture

First Order Markov Model:
- Overall experiment seems unsuccessful, however, the First Order Markov Model could still be utilized as a possible initialization pattern for the LM Head of a model.

Sankey Diagrams:
- Model may have different underlying processes to compute tokens and hidden representations that are part of the input sequence w.r.t. ones that are generated.
- A possible way to check that would be to run the model to generate some text and then take the generated output (together with the input) and feed it to the model as a new input. 
- Problems that may be observed while performing this process are:
    - not looking at residuals
    - text is too short
    - text is too close to the input tokens generated
- When using gradients, gradients are not normalized w.r.t. the other info of the model.

### Tasks

First Order Markov Model:
- [ ] Compare KL divergence on softmax results.
- [ ] Increase number of samples to improve significance.
- [ ] Possibly quantify the expected KL divergence to have a baseline.

Sankey:
- [ ] Investigate re
- [ ] Add residuals to the visualization (gradual implementation: first assume they contribute 50% and observe the graph, then insert their actual values).
- [ ] Add node color and fix order.
- [ ] Replicate visualization using gradients.
- [ ] Implement visualization using Dash.

Embeddings Analogies:
- [ ] Try different sizes of the same model
- [ ] Try comparing models by their embeddings size.