# Eliciting Latent Predictions from Transformers with the Tuned Lens (November 2023)

## Topics
Introduction of the tuned lens as an improvement on the logit lens, using a learned affine transformation and a learned bias.

## Abstract
We analyze transformers from the perspective of iterative inference, seeking to understand how model predictions are refined layer by layer.
To do so, we train an affine probe for each block in a frozen pretrained model, making it possible to decode every hidden state into a distribution over the vocabulary.
Our method, the tuned lens, is a refinement of the earlier "logit lens" technique, which yielded useful insights but is often brittle.
We test our method on various autoregressive language models with up to 20B parameters, showing it to be more predictive, reliable and unbiased than the logit lens.
With causal experiments, we show the tuned lens uses similar features to the model itself.
We also find the trajectory of latent predictions can be used to detect malicious inputs with high accuracy.
All code needed to reproduce our results can be found at https://github.com/AlignmentResearch/tuned-lens.

## Contents

### Research Questions

#### Introduction of the tuned lens, proof of its soundness and possible applications

We find that **tuned lens** predictions have substantially *lower perplexity* than **logit lens** predictions, and are more *representative* of the *final layer distribution*.

We use the **tuned lens** to gain *qualitative* insight into the computational process of pretrained language models, by examining how their *latent predictions* evolve during a *forward pass*.

#### Introduction of the Causal Basis Extraction technique as a general interpretability tool for transformer analysis

We show that the *most influential* features on the *tuned lens output* are also *influential* on the *model* itself.
To do so, we introduce a novel algorithm called **Causal Basis Extraction** (**CBE**) and use it to locate the *directions* in the *residual stream* with the *highest influence* on the *tuned lens*.

We then *ablate* these *directions* in the corresponding model *hidden states*, and find that these features tend to be *disproportionately influential* on the *model output*.

### Approach

We build on the "*logit lens*", an *early exiting* technique that directly decodes hidden states into vocabulary space using the model's pretrained *unembedding matrix* $W_U$.
We find the *logit lens* to be *unreliable*, failing to elicit plausible predictions for some models.
Even when the logit lens appears to work, its outputs are *hard to interpret* due to *representational drift*: features may be represented *differently* at different layers of the network.

To address the shortcomings of the *logit lens*, we introduce the **tuned lens**.
We train $L$ *affine transformations*, one for each layer of the network, with a *distillation loss*: transform the hidden state so that its image under the *unembedding* matches the *final layer logits* as closely as possible.
We call these transformations **translators** because they "*translate*" representations from the *basis* used at one layer of the network to the *basis* expected at the *final layer*.
Composing a *translator* with the pretrained unembedding yields a *probe* that maps a *hidden state* to a *distribution over the vocabulary*.

**Logit lens** for hidden states $\vec{h}$ at layer $\ell$:
$$LogitLens(\vec{h}_{\ell}) = LayerNorm[\vec{h}_{\ell}]W_U$$

**Tuned lens** for hidden states $\vec{h}$ at layer $\ell$, given $A_\ell$ (change of basis matrix) and $\vec{b}_\ell$ (bias) learned *affine transformation* given change:
$$TunedLens(\vec{h}_{\ell}) = LogitLens(A_\ell\vec{h}_{\ell} + \vec{b}_\ell)$$

*Translators* are trained by *minimizing the KL-divergence* between the *tuned lens logits* and the *final logits* (this can be viewed as a **distillation loss**, using the final layer distribution as a *soft label*):
$$\argmin{\mathbb{E}_x\left[D_{\mathcal{KL}}(f_{>\ell}(\vec{h}_\ell)||TunedLens(\vec{h}_\ell))\right]}$$
where $f_{>\ell}(\vec{h}_\ell)$ refers to the rest of the transformer after layer $\ell$.

To explore whether the *tuned lens* finds *causally relevant* features, we will assess two desired **properties**:
1) *Latent directions that are important to the tuned lens should also be important to the final layer output*.
2) *These latent directions should be important in the same
way for both the tuned lens and the model.
We will call this property stimulus-response alignment*.

To test **Property 1**, we first need to find the *important directions* for the *tuned lens*.
**Amnesic probing** provides one way to do this by seeking a direction whose erasure *maximally degrades* a model's *accuracy*.
We would like to find *many* such directions.
To do so, we borrow intuition from *PCA*, searching for additional directions that also degrade accuracy, but which are *orthogonal* to the *original amnesic direction*.
This leads to a method that we **call Causal Basis Extraction** (**CBE**), which finds the the *principal features* used by a model.

Given a function $f$ that maps *latent vectors* $\vec{h} \in \mathbb{R}^d$ to logits $\vec{y}$, **CBE** works by extracting a basis $B = (\vec{v}_1,...,\vec{v}_k)$ ordered according to $\Sigma = (\sigma_1,...,\sigma_k)$ (with $k << d$) iteratively, where for each iteration we search $\vec{v}_i$ s.t.:
$$\vec{v}_i = \argmax_{||\vec{v}||_2 = 1}{\sigma(\vec{v};f)} \;\;\; s.t. \; \left<\vec{v}, \vec{v}_j\right> = \vec{0}, \;\; \forall j<i$$
Where, given $r(\vec{h}, \vec{v})$ erasure function which removes information along the span of $\vec{v}$ from $x$, we have that:
$$\sigma(\vec{v};f) = \mathbb{E}_{\vec{h}}\left[D_{\mathcal{KL}}\left(f(\vec{h}_\ell)||f(r(\vec{h}, \vec{v}))\right)\right]$$

We check that the $k$ *directions* found with **CBE** on the *tuned lens* are also important for the *model*.
That is given the model $\mathcal{M}$, an i.i.d. sample of the input sequences $\vec{x}$ and for each $\vec{v}_i \in B$:
$$\mathbb{E}_{\vec{x}}\Bigl[D_{\mathcal{KL}}\bigl(\mathcal{M}(\vec{x})||\mathcal{M}_{>\ell}\left(r(\mathcal{M}_{\le \ell}(\vec{x}), \vec{v}_i)\right)\bigr)\Bigr]$$

We now turn to **Property 2**.
Intuitively, for the interventions, deleting an *important direction* should have the same effect on the *model*'s output distribution and the *tuned lens*' output distribution.

We use the "**Aitchison similarity**" between the *outputs* of the *tuned lens* and the *model* to measure the **stimilus-response alignment**.
The "**Aitchison similarity**" is *cosine similarity-like* metric based on the *weighted inner product* and *subtraction* operations inside the *Aitchison geometry*, which turns the *probability simplex* into a *vector space*.

### Experiments

#### Tuned Lens

- One problem with the *logit lens* is that, if transformer layers learn to *output residuals* that are *far from zero* on average, the input to LogitLens may be *out-of-distribution* and yield *nonsensical* results.
Our first change to the method is to replace the *summed residuals* with a *learnable constant* value $\vec{b}_\ell$ instead of *zero*.
- Another issue with the *logit lens* is that transformer hidden states often contain a *small number* of *very high variance* dimensions, and these "*rogue
dimensions*" tend to be distributed *unevenly* across layers.
If LogitLens relies on the *presence* or *absence* of particular *outlier* dimensions, the *perplexity* of *logit lens* predictions might be *spuriously high*.
Even when controlling for *rogue dimensions*, we observe a *strong* tendency for the *covariance matrices* of hidden states at different layers to *drift apart* as the number of layers *separating* them increases.
One simple, general way to correct for *drifting covariance* is to introduce a *learnable change of basis matrix* $A_\ell$, which learns to map from the *output space* of layer $\ell$ to the *input space* of the *final layer*.
##### Findings
- We find that the *tuned lens* resolves the problems with the *logit lens*: it has *significantly lower bias*, and* much lower perplexity* than the *logit lens* across the board.

#### Transferability across layers/models

- We find that *tuned lens translators* can usually *zero-shot transfer* to *nearby layers* with only a *modest* increase in *perplexity*.
Specifically, we define the* transfer penalty* from layer $\ell$ to $\ell'$ to be the *expected increase in cross-entropy loss* when evaluating the *tuned lens* translator trained for layer $\ell$ on layer $\ell'$.
- Same process is repeated for testing transferring *tuned lense* models across *models*.
Here a transferred lens is one that makes use of the *fine-tuned models*' *unembedding*, but simply copies its *affine translators* from the lens trained on the *base model*.
##### Findings
- Overall, *between-layer transfer penalties* are quite *low*, especially for *nearby layers*.
We notice that *transfer penalties* are *strongly negatively correlated* with *covariance similarity* (Spearman $\rho=−0.78$).
*Transfer penalties* are *higher* when training on a layer with the *outlier dimensions* and testing on a layer *without* them, **than the reverse**.
- We find that lenses trained on a base model *transfer well* to *fine-tuned versions* of that *base model*, with *no additional training* of the lens.
Transferred lenses substantially outperform the Logit Lens and *compare well* with lenses trained *specifically* on the finetuned models.
- Model finetuning *minimally effects* the representations used by the tuned lens.
This opens applications of the tuned lens for *monitoring changes* in the *representations* of a module *during fine-tuning* and *minimizes* the need for practitioners to train lenses on new fine-tuned models.

#### Measuring causal fidelity

- **Property 1** test
- **Property 2** test
##### Findings
- In accordance with **Property 1**, there is a *strong correlation* between the *causal influence* of a feature on the *tuned lens* and its *influence* on the *model* (Spearman $\rho = 0.89$).
Importantly, we **don't observe** any features that are *influential* in the tuned lens *but not in the model*.
The model is somewhat more "*causally sensitive*" than the tuned lens: even the *least influential* features never have an influence under $2 \times 10 − 3 \; bits$.
- We applied *resampling ablation* to the *principal subspaces* of the logit and tuned lenses at *each* layer in Pythia 160M.
Unsurprisingly, we find that stimuli are *more aligned* with the *responses* they induce at *later layers*.
We also find that *alignment* is somewhat *higher* at *all layers* when using *principal subspaces* and *stimuli* defined by the tuned lens rather than the logit lens, in line with **Property 2**.

#### Detecting Prompt Injections

- We hypothesize that the *prediction trajectory* of the tuned lens on *anomalous inputs* should be different from the *trajectories* on *normal inputs*, and that this could be used to *detect anomalous inputs*.
To test this, we focus on **prompt injection attacks** (these attacks usually tell the model to "ignore previous instructions" and instead follow instructions crafted by the attacker).
- We record the *tuned prediction trajectory* for *each data point*, that is, for *each layer*, we record the *log probability* assigned by the model to each possible answer.
We then *flatten* these *trajectories* into *feature vectors* and feed them into *two standard outlier detection algorithms*.
- We *fit* each *anomaly detection model* exclusively on prediction trajectories from *normal prompts* **without** *prompt injections*, and *evaluate* them on a *held out test set* containing **both** *normal* and *prompt-injected trajectories*.
- We use the *Simplified Relative Mahalanobis* (*SRM*) distance as a *baseline* in our experiments.
##### Findings
- Our tuned lens *anomaly detector* achieves *perfect* or *near-perfect* AU-ROC on *five tasks*; in contrast, the same technique using logit lens has *lower performance* on most tasks.
On the other hand, the *SRM baseline* does *consistently well*, the tuned lens **only** *outperforms* it on *one* task.

#### Measuring Example Difficulty

- Some early exiting strategies are based on the observation that "*easy*" examples require *less computatio*n to classify than "*difficult*" examples.
We propose to use the tuned lens to *estimate example difficulty in pretrained transformers*, without the need to fine-tune the model for early exiting.
- We evaluate the model's *zero-shot performance* on twelve multiple-choice tasks.
For each checkpoint, we store the *top 1 prediction* on every individual example, allowing us to compute the iteration learned.
We then use the tuned lens on the final checkpoint, eliciting the *top 1 prediction* at *each layer of the network* and computing the *prediction depth* for every example.
- As a *baseline*, we also compute *prediction depths* using the *logit lens*.
Finally, for each task, we compute the *Spearman rank correlation* between the *iteration learned* and the *prediction depth* across all examples.
##### Findings
We find a *significant positive correlation* between the *iteration learned* and the *tuned lens prediction depth* on **all** tasks we investigated.
Additionally, the *tuned lens* prediction correlates *better* with *iteration learned* than its *logit lens* counterpart in 8 out of 11 tasks, sometimes dramatically so.

### Models

- GPT-2
- GPT-Neo
- BLOOM 560M
- OPT 125M 
- Pythia
- LLaMA 13B
- Vicuna 13B

### Datasets

- Respective model pretraining validation sets (when available)
- The Pile (validation set)
- Benchmarks: ARC-Easy, ARC-Challenge, BoolQ, MC TACO, MNLI, QNLI, QQP, SciQ, SST-2, LogiQA, PiQA, RTE, WinoGrande 
