# Do Llamas Work in English? On the Latent Language of Multilingual Transformers (February 2024)

## Topics
Investigation on the transformers' internal hidden representation space bias towards english in the context of translation-oriented tasks.

## Abstract
We ask whether multilingual language models trained on unbalanced, English-dominated corpora use English as an internal pivot language — a question of key importance for understanding how language models function and the origins of linguistic bias.
Focusing on the Llama-2 family of transformer models, our study uses care fully constructed non-English prompts with a unique correct single-token continuation.
From layer to layer, transformers gradually map an input embedding of the final prompt token to an output embedding from which next-token probabilities are computed.
Tracking intermediate embeddings through their high-dimensional space reveals three distinct phases, whereby in termediate embeddings (1) start far away from output token embeddings; (2) already allow for decoding a semantically correct next token in middle layers, but give higher probability to its version in English than in the input language; (3) finally move into an input-language-specific region of the embedding space.
We cast these results into a conceptual model where the three phases operate in "input space", "concept space", and "output space", respectively.
Crucially, our evidence suggests that the abstract "concept space" lies closer to English than to other languages, which may have important consequences regarding the biases held by multilingual language models.
Code and data is made available here: https://github.com/epfl-dlab/llm-latent-language.

## Contents

### Research Questions

#### Does english pivoting occur implicitly when LLMs are prompted in non-English languages?

In the research community as well as the popular press, many seem to assume that the answer is yes.
In the one hand, implicitly using *English* as an *internal pivot* could *bias* LLMs toward *Anglocentric patterns* that could predispose the model to certain *linguistic elements* while also shaping more profound behaviors related to *emotional stance* or *temporal reasoning*.
On the other hand, if LLMs do not use English as a *pivot*, it raises questions of *how* else they manage to work so *remarkably well* even in *low-resource* languages.

We find that applying the "*unembedding*" operation *prematurely* in *intermediate*, non-final layers (*logit lens*) already decodes a *contextually appropriate* token *early on*, giving us a (limited) glimpse at the model’s otherwise hard-to-interpret numerical *internal state*.
Exploiting this fact, we carefully devise *prompts* that allow us to determine whether a logit-lens-decoded token is *semantically correct* and to *what language* it belongs.

Tracking *language probabilities across layers*, we observe that **no** *contextually appropriate tokens* are decoded in the *first half* of layers, followed by a *sudden shift of probability mass* onto the *English* version (“flower”) of the correct next token, and finally a shift to the correct next token in the *target language* ("花").

Expanding on this first evidence of *English* as an *internal pivot* language, we analyze *latent embeddings* directly as **high-dimensional Euclidean points**, rather than via the logit lens.
This suggests that in middle layers, the transformer operates in an abstract "**concept space**" that is *partially orthogonal* to a language-specific “*token space*”, which is reached only in the *final layers*.
In this interpretation, the latent embeddings' proximity to English tokens observed through the *logit lens* follows from an English *bias* in concept space, rather than from the model first translating to English and then "*restarting*" its forward pass from there.

### Approach

Due to **RMS normalization**, all latents lie on a $d$-dimensional *hypersphere* of radius $\sqrt{d}$.

Although the *logit lens* allows us to map *latent vectors* to *token distributions*, we still require a mapping from *token distributions* to *languages*.
Doing so in general is difficult as many tokens are *ambiguous* with respect to *language*.
To circumvent this issue, we construct *prompts* $x_1,...,x_n$ where the correct *next token* $x_{n+1}$ is (1) **obvious** and (2) **can be unambiguously attributed to one language**.

To enable *unambiguous language attribution*, we construct a *closed set of words* per language.
We scan Llama-2’s vocabulary for *single-token Chinese words* (mostly
nouns) that have a *single-token English translation*.
For each other language (German, French, and Russian) use the *same words*, but discard those that share a *token prefix* with the English version, as this would render language detection ambiguous.

To investigate a hypothetical *pivot language* inside Llama-2, we apply the *logit lens* to the latents $h^{(j)}_n$ corresponding to the *last input token* $x_n$ for each layer $j$, obtaining one *next-token* distribution $P(x_{n+1}|h^{(j)}_n)$ per layer.
Our prompts are specifically designed such that an *intermediate next-token distribution* lets us estimate the probability of the *correct next word* in the input language as well as English.
Since we specifically select single-token words in Chinese (ZH) as well as English (EN), we can simply define the probability of language $\ell \in \{ZH , EN\}$ as the probability of the next token being $\ell$'s version $t_\ell$ of the correct single-token word: $P(lang=\ell|h^{(j)}_n) := P(x_{n+1}=t_\ell|h^{(j)}_n)$.

### Experiments

#### Probabilistic View

The *logit lens* gives us one set of language probabilities per input prompt and layer.

- Tasks:
    1) **Translation task**: Here the task is to *translate* the preceding *language A* word to *language B* (both not being English).
    We show the model four words with their correct translations, followed by a fifth word without its translation, and let the model predict the next token.
    2) **Repetition task**: We task the model to simply *repeat the last word*, instead of translating it.
    3) **Cloze task**: We consider a *cloze test*, where the model must predict a masked word in a sentence.
    Given a target word, we construct an English sentence starting with the word by prompting GPT-4, mask the target word, and translate the sentence to the other languages.
    To construct prompts, we sample two demonstrations from the remaining words.
- A transformer's forward computation for a given *final input* token $x_n$ can now be visualized by connecting the *2D-projected embeddings *of the latents $h^{(j)}_n$ in subsequent layers $j$.
##### Findings
- On the *translation* and *cloze* tasks a consistent picture emerges across model sizes.
Neither the correct Chinese token nor its English analog garner **any** noticeable *probability mass* during the *first half of layers*.
Then, around the *middle layer*, English begins a *sharp rise* followed by a decline, while Chinese *slowly grows* and, after a *crossover* with English, *spikes on the last five layers*.
- On the *repetition* task, Chinese already *rises alongside* English.
This is in contrast to all other languages, where English *rises first*.
- Regarding the *entropy* of the *full next-token distribution*, we again observe a *consistent pattern* across tasks and model sizes: *high entropy in the first half of layers*, while both $P(lang = ZH)$ and $P(lang = EN)$ are *close to zero*, followed by a *sharp drop* at the same time that $P(lang = EN)$ *rises*.
From there on, *entropy remains low*, with a *slight rebound* as probability mass shifts from English to Chinese.
- With $32,000 \approx 2^{15}$ tokens in the vocabulary, the *early entropy* of around $14$ bits implies a *close-to-uniform next-token distribution* (around $15$ bits).
- Regarding the *forward computation paths*: an English and a Chinese *token cluster* emerges, suggesting that the same latent also gives *high probability* to an *entire language*, in addition to the language-specific version of the correct next token.
- Paths *first pass through* the English cluster, and *only later reach* the Chinese cluster.
Taken together, the emerging picture is that, when translating a German word to Chinese, Llama-2 takes a "detour" through an English subspace.

#### Geometric View

- *Output token embeddings* (rows of the é $U$) and latents $h$ cohabitate the same $d$-dimensional *Euclidean space*.
In fact, due to *RMS-normalization*, latents by construction *live on a hypersphere* of radius $\sqrt{d} \approx 90.1$.
Additionally, by analyzing the $2$-norm of *output token embeddings*, we find that the latter also approximately lie on a sphere, of radius 1.52.
- Token embeddings occupy their sphere **unevenly**.
- To build intuition, first consider a *hypothetical extreme case* where tokens lie in a proper subspace ("*token subspace*") of the full $d$-dimensional space (even though, empirically, $U$ has rank $d$).
If a latent $h$ has a component *orthogonal* to the *token subspace*, it includes *information* that is **irrelevant** for p*redicting the next token* based on $h$ alone (since logits are scalar products of latent and token vectors).
- The *orthogonal component* can still be important for the computations carried out by *later layers*.
But the *logit lens*, which decodes *latents* into tokens prematurely in intermediate layers, will be *blind* to the *orthogonal component*.
A latent $h$'s *angle* with the "*token subspace*" thus measures how much of $h$ is irrelevant for immediately predicting the next token.
- Concretely, we consider the *mean squared cosine* between $h$ and the *token embeddings* (rows of $U$) to capture how much of $h$'s "**energy**" translates into *logit scores*.
##### Findings
- As a function of layer, *root mean squared token energy* is *low* (around 20%) and *mostly flat before layer 70*, when it suddenly *spikes*, just when next-token predictions switch from English to Chinese.
- In sum, reveals three phases:
    1) **Phase 1** (layers 1–40): *High entropy* (14 bits, nearly uniform), *low token energy*, *no language dominates*.
    2) **Phase 2** (layers 41–70): *Low entropy* (1–2 bits), *low token energy*, *English dominates*.
    3) **Phase 3** (layers 71–80): *Low entropy*, *high token energy* (up from 20% to 30%), *Chinese dominates*.
- **Phase 1** is focused on building up a *better feature representation* for the current token from its input embedding, by dealing with *tokenization* issues.
This phase is not yet directly concerned with predicting the *next token*, with latents remaining *largely orthogonal* to output token space (low token energy), leading to *small dot products* between latents and output token embeddings, and thus to *high entropy*.
- In **Phase 2**, latents live in an abstract "*concept space*", which, unlike in *Phase 1*, is *no more orthogonal* to the output token space.
Rather, latent "*concept embeddings*" are closer to those output token embeddings that can express the respective concept, leading to *low entropy*.
Among the concept-relevant tokens, English variants lie *closer* to the concept embedding than non-English variants (due to the model's overwhelming exposure to English during training), leading to higher probabilities for English than Chinese tokens.
Despite the correlation between concept and token embeddings, concept embeddings also carry *much information* that goes **beyond** output tokens, leading to a *still-low token energy*.
- In **Phase 3**, the model maps *abstract concepts* to *concrete words/tokens* in the target language.
Information that is *irrelevant* for next-token prediction is *discarded*, leading to a *spike* in *token energy*.

### Models

- LLaMA-2 7B 13B 70B

### Datasets

--