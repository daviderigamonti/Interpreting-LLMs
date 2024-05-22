# All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality (November 2021)

## Topics
Analysis of the presence, influence and solutions for rogue dimensions inside the embeddings space.

## Abstract
Similarity measures are a vital tool for understanding how language models represent and process language.
Standard representational similarity measures such as cosine similarity and Euclidean distance have been successfully used in static word embedding models to understand how words cluster in semantic space.
Recently, these measures have been applied to embeddings from contextualized models such as BERT and GPT-2.
In this work, we call into question the informativity of such measures for contextualized language models.
We find that a small number of rogue dimensions, often just 1-3, dominate these measures.
Moreover, we find a striking mismatch between the dimensions that dominate similarity measures and those which are important to the behavior of the model.
We show that simple postprocessing techniques such as standardization are able to correct for rogue dimensions and reveal underlying representational quality.
We argue that accounting for rogue dimensions is essential for any similarity-based analysis of contextual language models

## Contents

### Research Questions

#### Informativity analysis of similarity measures in the embedding space

Recent work which probes the *representational geometry* of *contextualized embedding spaces* using *cosine similarity* has found that *contextual embeddings* have several *counterintuitive* properties.
1) Word representations are highly **anisotropic**: *randomly* sampled words tend to be *highly similar* to one another when measured by *cosine similarity*.
2) Embeddings have extremely **low self-similarity**: In later layers of transformer-based language models, random words are *almost as similar* to one another as instances of the same word in different contexts.

We find that these measures are often dominated by *1-5 dimensions* across all the *contextual language models* we tested, regardless of the specific *pretraining objective*.
It is this *small subset* of dimensions which drive *anisotropy*, *low self-similarity*, and the apparent drop in *representational quality* in later layers.

These dimensions, which we refer to as **rogue dimensions** are centered *far* from the *origin* and have disproportionately *high variance*.
The presence of **rogue dimensions** can cause *cosine similarity* and *Euclidean distance* to rely on less than *1%* of the embedding space.
Moreover, we find that the *rogue dimensions* which dominate *cosine similarity* do not likewise dominate *model behavior*, and show a strong correlation with *absolute position* and *punctuation*.

#### Are there alternative similarity measures and/or solutions to account for rogue dimensions in the embedding space?

We show that these **rogue dimensions** can be accounted for using a trivially *simple transformation* of the embedding space: standardization.
Once applied, *cosine similarity* more closely reflects human word similarity judgments, and we see that *representational quality* is preserved across all layers rather than degrading/becoming task-specific.

### Approach

Previous words showed that contextual embedding spaces are highly *anisotropic*, meaning that the *contextual representations* of any two tokens are expected to be highly similar to one another.

Given a function $CC_i(u,v) = \frac{u_iv_i}{||u||||v||}$ that describes the contribution of dimension $i$ to the total of a *cosine similarity* between $d$ dimensions we can write the contribution of dimension $i$ to the **approximated** (over a random sample $S$ of $n$ token pairs from corpus $\mathcal{O}$: $S = \bigl\{\{x_i, y_i\}\bigr\}_{i=1}^n \sim \mathcal{O}$ ) **anisotropy** $\hat A (f_\ell)$ in layer $\ell$ of model $f$ as:
$$CC(f_\ell^i) = \frac{1}{n} \cdot \sum_{\{x_\alpha, y_\alpha\} \in S}{CC_i(f_\ell(x_\alpha),f_\ell(y_\alpha))}$$

This results in $\hat A (f_\ell) = \sum_i^d{CC(f_\ell^i)}$.

We also investigate whether *standard similarity measures* are still *informed* by the *entire embedding space*, or if **variability** in the measure is also driven by a *small subset* of dimensions.
For example, it could be the case that some dimension $i$ has a large, but roughly constant activation across all tokens, meaning large $\mathbb{E}[CC(f_\ell^i)]$ and low $Var[CC(f_\ell^i)]$.
In this case, we would be adding a large constant to *cosine similarity* and the *average cosine similarity* would be driven toward 1 by dimension $i$, but any changes in *cosine similarity* would be driven by the rest of the embedding space, meaning *cosine similarity* would provide information about the entire representation space, rather than a single dimension.
In the opposite case, dimension $i$ would not appear to make the space anisotropic, but would still drive variability in *cosine similarity*.
Ultimately, we are interested in whether changes in a *similarity measure* reflect changes in the *entire embedding space*.

To measure this contribution we use the **Pearson correlation** between $C(S) = \cos\limits_{x,y \in S}{(f_\ell(x), f_\ell(y))}$ and $C'(S) = \cos\limits_{x,y \in S}{(f'_\ell(x), f'_\ell(y))}$ where $f_\ell(x)$ is the function that maps a token $x$ to its $d$-dimensional representation in layer $\ell$ of model $f$, while $f'_\ell(x)$ is the similar, but the token $x$ is mapped onto the $d$-dimensional space with its top-$k$ dimensions removed:
$$r = corr[C(S), C'(S)]$$

We also measure the *influence* of *individual dimensions* on *model behavior* through an *ablation* study.
We measure how the *distribution changes* after ablation using **KL divergence** between the *ablated model distributio*n (setting dimension $i$ of layer $\ell$ to zero) and the *unaltered reference distribution* $P_f(s)$:
$$I(i, \ell, f) = \frac{1}{n}\sum_{s \in S}^{n}{D_{\mathcal{KL}}[P_f(s) || P_f(s|f_\ell^i(s) = 0)]}$$

There are several simple *postprocessing methods* which can correct for how *rogue dimensions* influence *similarity metrics*.
1) **Standardization**: We have observed that a small subset of dimensions with *means far from zero* and *high variance* completely dominate *cosine similarity*.
A straightforward way to adjust for this is to subtract the *mean vector* and divide each dimension by its *standard deviation*, such that each dimension has $\mu_i = 0$ and $\sigma_i = 1$.
2) **All-but-the-top**: Following from similar observations (a nonzero common mean vector and a small number of dominant directions) in static embedding models, others proposed subtracting the *common mean vector* and *eliminating the top few principle components* (they suggested the top $\frac{d}{100}$), which should *capture the variance* of the *rogue dimensions* in the model and make the space more *isotropic*.
3) **Spearman's $\rho$**: *Word embeddings* can be seen as $d$ observations from an $|\mathcal{O}|$-variate distribution, and use *Pearson correlation* as measure of *similarity*.
*Spearman correlation* is just *Pearson correlation* but between the *ranks* of embeddings, rather than their values.
Thus *Spearman correlation* can also be thought of as a *postprocessing technique*, where instead of standardizing the space or removing the top components, we simply transform embeddings as $x' = rank(x)$.
Spearman's $\rho$ is *robust to outliers* and thus will not be dominated by the *rogue dimensions* of contextual language models.
Unlike standardization and all-but-the-top, Spearman correlation requires *no computations over the entire corpus*.

### Experiments

#### Anisotropy

- We compute the *average cosine similarity contribution* $CC(f_\ell^i)$ for *each dimension* in *all layers* various models.
We then *normalize* by the *total expected cosine similarity* $\hat A (f_\ell)$ to get the proportion of the total expected cosine similarity contributed by each *dimension*.
- All models are of dimensionality $d = 768$ and have *12 layers*, plus one static *embedding* layer.
We also include two 300 dimensional *non-contextual models* (Word2Vec and GloVe) for comparison.
##### Findings
- The *static models* Word2Vec and GloVe are *relatively isotropic* and are **not** dominated by any *single* dimension.
Across all transformer models tested, a *small subset of rogue dimensions* dominate the *cosine similarity* computation, especially in the more **anisotropic** *final layers*.
The dimensions which drive *anisotropy* are centered *far* from the *origin* relative to other dimensions.
- One implication of *anisotropy* is that the embeddings occupy a *narrow cone* in the *embedding space*, as the *angle* between any two word embeddings is *very small*.
However, if *anisotropy* is driven by *a single dimension* (or a small subset of dimensions), we can conclude that the cone lies along a *single axis* or within a *low dimensional subspace*, rather than being a global property across all dimensions.
- *Anisotropy* of the embedding space is an *artifact* of *cosine similarity*'s high *sensitivity* to a small set of *outlier dimensions* and is **not a global property** of the space.

#### Informativity of Similarity Measures

- For this experiment, we compute $r^2$ for *all layers* of *all models*.
We remove the top $k = {1, 3, 5}$ dimensions, where dimensions are ranked by $CC(f_\ell^i)$.
##### Findings
- We find that in the *static embedding models* and the *earlier layers* of each *contextual model*, *no single dimension* or *subset of dimensions* drives the *variability* in *cosine similarity*.
By contrast, in *later layers*, the *variability* of *cosine similarity* is driven by just *1-5 dimensions*.
- Token pairs which are *similar* to one another in the *full embedding* space are *drastically different* from the pairs which are *similar* when just a handful of dimensions are *removed*.
- Small subset of dimensions in later layers seem to drive the *cosine similarity* of *randomly sampled* words toward 1, but this subset also drives the *variability* of the measure.
This result effectively renders* cosine similarity* a measure over *1-5 rogue dimensions* rather than the *entire embedding space*.

#### Behavioral Influence of Individual Dimensions

- To measure the *importance* of each dimension to *model behavior*, we compute $I(i, \ell, f)$ for the last *4 layers* of each model over 10k distributions.
##### Findings
- In all models, we see that the *dimensions* which dominate *cosine similarity* do **not** likewise dominate *model behavior*.
- While *ablating rogue dimensions* often *alters* the language modeling *distribution* more than *ablating non-rogue dimensions*, we emphasize that there is **not** a *one-to-one correspondence* between a dimension's influence on *cosine similarity* and its influence on *language modeling behavior*.

#### Postprocessing

- We evaluate whether the *cosine similarities* between *word pairs* align more *closely* with *human similarity judgments* after *postprocessing*.
We evaluate this using word *similarity/relatedness* judgment datasets where examples consist of a pair of words and a corresponding *similarity rating* averaged over several human annotators. 
##### Findings
- *Postprocessing* aligns the *embedding space* more closely to *human similarity judgments* across almost *all layers* of *all models*.
We found that **standardization** was the *most successful postprocessing method*, showing consistent improvement over the original embeddings in all but the early layers of BERT.
- **All-but-the-top** was *generally effectiv*e but highly *dependent* on the *number of components removed*.
- Simply **subtracting the mean vector** (common factor between *standardization* and *all-but-the-top*) also yielded *substantial gains* in most models.
- *Converting embeddings into ranks* (**Spearman correlation**) also resulted in *significantly stronger correlations* with human judgments in *all layers* of *all models*, though the correlation was often *weaker* than *standardization* or *all-but-the-top*.
- Previous work has suggested that the *final layers* of transformer language models are *highly task-specific*.
Our findings suggest that linguistic representational quality is actually *preserved* in the *final layers* but is *obscured* by a *small handful* of *rogue dimensions*.
After simple *postprocessing*, *later layers* of the model correlate just as well, if not better than *intermediate layers* with human similarity judgments.

### Models

- GPT-2
- BERT
- RoBERTa
- XLNet
- Word2Vec
- GloVe

### Datasets

- English Wikipedia
- RG65, WS353, SimLex999, SimVerb3500