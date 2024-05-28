# Interpreting GPT: the logit lens (August 2020)

## Topics
Introduction of the logit lens to look and interpret transformer intermediate representations.

## Abstract
--

## Contents

### Research Questions

#### Do inputs representations get discarded immediately in the transformers' internal latent spaces? If so, why?

In the **logit lens**, the early layers sometimes look like *nonsense*, and sometimes look like very *simple guesses* about the output.
They almost *never* look like the input.
Apparently, the model does **not** "k*eep the inputs around*" for a while and gradually process them into some intermediate representation, then into a prediction.
Instead, the inputs are *immediately* converted to a very different representation, which is smoothly refined into the final *prediction*.

1) Transformers are *residual networks*.
Every connection in them looks like $x + f(x)$ where $f$ is the learned part.
So the **identity** is very *easy* to learn.
This tends to keep things in the *same basis* across *different layers*, unless there's some reason to switch.
2) Transformers are usually trained with *weight decay*, which is almost the same thing as *L2 regularization*.
This encourages learned weights to have *small L2 norm*.
That means the model will try to "*spread out*" a computation across as many layers as possible (since the sum-of-squares is less than the square-of-sums).
Given the task of turning an input into an output, the model will generally prefer changing the input *bit by bit*.

1+2 are a good story if you want to explain why the same vector basis is used across the network, and why things change smoothly, but remember that the *input is discarded in such a discontinuous way*!
A *U-shaped pattern* would be expected, where the early layers mostly look like the input, the late layers mostly look like the output, and there's a gradual "*flip*" in the middle between the two perspectives.

#### Are some rare tokens "kept around" by the transformer in order to make their repetition easier? If so, why?

An intuitive guess is that the *rarity*, or (in some sense) "*surprisingness*," of the token causes early layers to *preserve it*: this would provide a mechanism for providing *raw access to rare tokens* in the later layers, which otherwise only be looking at more plausible tokens that GPT had guessed for the corresponding positions.

On the other hand, some *discordant* evidence is found.

### Approach

GPT's *probabilistic predictions* are a *linear function* of the **activations** in its final layer.
If one applies the same function to the *activations of intermediate GPT layers*, the resulting distributions make intuitive sense.
This "*logit lens*" provides a simple (if partial) interpretability lens for GPT's internals.
The *logit lens* focuses on what GPT "*believes*" after each step of processing, rather than how it updates that belief inside the step.

### Experiments

#### Rank view

- Visualize *top-1* guess at each layer.
- Visualize *rank* of the model's *final top-1 guess* (not the true token) at each layer.
##### Findings
- In most cases, *network's uncertainty* has drastically *reduced* by the middle layers.

#### KL-divergence and input discarding

- Comparing the *similarity* of two probability distributions through the **KL divergence**: by taking the *KL divergence* of the *intermediate probabilities* w.r.t. the *final probabilities*, we get a more *continuous* view of how the distributions *smoothly converge* to the model's output.
- Also plot the $KL(input||layer)$ instead of the $KL(output||layer)$ of the previous point and compare both against the rank view.
##### Findings
- Immediately, after the very *first layer*, the input has been *transformed* into something that looks more like the final output (47 layers layer) than it does like the input; after this one *discontinuous jump*, the distribution progresses in a much more smooth way to the final output distribution.
- There is still a *fast jump* in $KL(input||layer)$ after the input, but it's far *smaller* than the jump in $KL(output||layer)$ at the same point.
- Likewise, while *ranks jump quickly* after the input, they often stay relatively high in the context of a ~50K vocab, but in particular some tokens are "*preserved*" much more in this sense than others.
This is apparently *contextual*, not just based on the token itself.
- It's possible that the *relatively high ranks* (in the 100s or 1000s, but not the 10000s) of input tokens in many cases is (related to) the mechanism by which the model "*keeps around*" *rarer* tokens in order to copy them later.

#### Copying a rare token

- Sometimes it's clear that the next token should be a "*copy*" of an *earlier token*: whatever arbitrary thing was in that slot, spit it out again; if this is a token with *relatively low prior probability*, one would think it would be useful to "*keep it around*" from the input so later positions can look at it and copy it.
- Prompt: *Sometimes, when people say plasma, they mean a state of matter. Other times, when people say plasma*
##### Findings
- the model correctly predicts "*plasma*" at the last position, but only figures it out in the *very last layers*, apparently it is keeping around a representation of the token "*plasma*" with enough *resolution* to copy it,  but it only retrieves this representation *at the end*! (In the rank view, the rank of plasma is quite low until the very end)
- The *repetition* is directly visible in the input: "*when people say*" is copied *verbatim*.
If you just applied the rule "*if input seems to be repeating, keep repeating it*" you'd be good.
Instead, the model *scrambles away the pattern*, then *recovers*
it later through some other *computational route*.

### Models

- GPT-2

### Datasets

- Curated examples
