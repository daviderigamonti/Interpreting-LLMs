# Emergent and Predictable Memorization in Large Language Models (April 2023)

## Topics
Scaling analysis for predicting and limiting memorization in Large Language Models.

## Abstract
Memorization, or the tendency of large language models (LLMs) to output entire sequences from their training data verbatim, is a key concern for deploying language models.
In particular, it is vital to minimize a model's memorization of sensitive datapoints such as those containing personal identifiable information (PII).
The prevalence of such undesirable memorization can pose issues for model trainers, and may even require discarding an otherwise functional model.
We therefore seek to predict which sequences will be memorized before a large model's full train-time by extrapolating the memorization behavior of lower-compute trial runs.
We measure memorization in the Pythia model suite and plot scaling laws for forecasting memorization, allowing us to provide equi-compute recommendations to maximize the reliability (recall) of such predictions.
We additionally provide further novel discoveries on the distribution of memorization scores across models and data.
We release all code and data necessary to reproduce the results in this paper at https://github.com/EleutherAI/pythia

## Contents

### Research Questions

#### How is it possible to forecast if a model will memorize a specific datum contained in the training set?

Since some specific data points are far more *undesirable* for a model to *memorize*, such as **PII**, it would be desirable for engineers to be able to *predict* whether a model will *successfully avoid memorizing* such *harmful data*.

*Memorization* is typically cast as a form of *overfitting* or failure to *generalize* outside the training data distribution, distinct from "*good learning*" in some senses.

We consider the framework introduced by *Carlini et al.* grounded in **$k$-extractibility**: 
A string $s$ is said to be **$k$-extractible** if it
1) *exists* in the *training data*, and 
2) is *generated* by the *language model* by prompting with $k$ *prior tokens*.

#### Is it possible to extend a memorization forecast from a partially trained model or a smaller model to a bigger one?

Due to the substantial *cost* of training *large language models*, it is highly desirable to be able to *make predictions* about *model characteristics* before they are actually trained.
The literature on *scaling laws* has been successfully used to inform the *decision-making* of a variety of researchers at model *training-time* by allowing them to *generalize* the decisions made while investigating *smaller models* to inform the design of *larger models*.

Using *smaller* or *partial* model training runs to *inform* large model training runs is critical, because these runs provide a *cheap* method to inform training behavior for a given *corpus*, rather than training an entire large model from scratch.

Unfortunately, the discovery that the *memorization* of a specific training string by a *large language model* is **not** *reliably predicted* by either studying *smaller* language models or *partially trained* checkpoints, unless a *sizable fraction* of the *pretraining compute* of the target model is used.

### Approach

We evaluate the **memorization score**, defined as the *number of ordered matching tokens* between the model's *greedily* generated sequence $G$ and the dataset's *true* continuation of a sequence $S \in D$ on a given prompt:
$$score(M,N) = \frac{1}{N}\sum_{i}^{N}{\mathbb{I}(S_{M+i} = G_{M+i})}$$

Where $N$ is the *length of the true continuation* and greedily generated sequence, and $M$ is the *length of the prompt*.
A memorized or extractable sequence has a memorization score of 1.

We can treat the fact of a *smaller* model memorizing a certain sequence as a "*prediction*", which compared against the *ground truth* allows us to calculate *classification metrics* such as *precision* and *recall*.
In this case, *precision* tells us how many of the sequences memorized by the *smaller* model are *also memorized* by the *larger* model.
*Recall* conveys the percentage of sequences memorized by the *larger* model that are *also memorized* by the *smaller* model.
The same framing can also be applied when analyzing *across time during training*.
We are primarily interested in assessing the *recall* of the predictors and will tolerate a *low precision* if it comes with a *high recall*.

### Experiments

#### Memorization Across Scales

- To evaluate how productive training *small* models can be for the purpose of predicting which *data points* will be *memorized* by *large* models, we *subset* our data to the sequences with a *memorization score* of 1 (meaning all $N$ = 32 target tokens were produced accurately by the smaller model). 
Then, we look at the *correlations* between each *pair* of *fully-trained* model *sizes* for which sequences are *memorized*.
We also measure *precision* and *recall* of *fully-memorized sequences*.
##### Findings
- We see a *sharp decline* in *correlation* between which sequences are *memorized* by *smaller* models and the *larg* models as the *gap* between the model sizes *increases*.
Unfortunately, we find that these *low correlation scores* cause the set of sequences memorized by small models to have very *poor predictive power* in terms of what sequences will be *memorized* by a larger model.  
- Although the *precision* is *high* for *all models*, we are more interested in achieving a *high recall* than a *high precision*.
The recall is incredibly low across the board.
Our findings suggest that using *smaller* model runs to *forecast* the *memorization* of *larger* models is **not accurate**.

#### Memorization Across Training

- We wish to determine if, by testing *memorization behavior* after *partially completing* a *training run*, an engineer can achieve a *reliable* signal about whether *undesirable* portions of the training data are *memorized* and if so to *abort* a training run *early*.
##### Findings
- Unfortunately, we continue to find *largely negative results*, but hope that future research with better techniques for predicting memorization might vindicate this idea.
- We see that the *earliest* *intermediate* checkpoints we test **do not** exhibit the *high recall* that is desirable.
We thus observe that using *intermediate checkpoints* of a model run to predict memorization is not a *silver bullet*: it is still the case that *precision* remains *high* throughout models, but *recall* is *low* for *all predictors* that use significantly less compute than the final model's cost.
Therefore, in this setting as well, it is *easier* to guarantee a sequence will be *memorized* through such extrapolations *rather than not*.

### Models

- Pythia

### Datasets

- The Pile
- ROOTS
