# Notes

Plots should be presented following a reasoning:
1) Research question
2) Theoretical explanation of subject and experiment contextualization (why we do what we do and what we would like to see)
3) Detailed explanation of the experiment (do not explain code line-by-line, but from the theoretical point of view)
4) Plots and results

# Question 1

QUESTION 1: Does the input embedding contain facts?

• EXPERIMENT 1: Check performance of input embedding on dataset of
analogies with input embedding for various size Mistral models. (May
also show results for the output decoder.)

https://arxiv.org/pdf/1901.09813.pdf

## Text

The first experiment aims at understanding if text embeddings of LLMs still encompass some degree of semantic factuality.
Embedding representations of text for early transformer models are known to have interesting spatial properties deeply correlated with the semantic properties of the embedded words \cite{DBLP:journals/corr/abs-2009-11226, DBLP:conf/icml/AllenH19}.
However, with the advent of new models that are bigger and more complex than ever, we would like to assess if the semantic properties inherently tied with the geometry of embeddings still hold true.


To verify these claims we opted for the classic word analogy task \cite{DBLP:conf/icml/AllenH19, TODO}, which given four words ($w_a$, $w_{b}$, $w_c$ and $w_d$) that follow an analogy relationship of the type $w_a : w_b = w_c : w_d$, by exploiting the linear properties of text embeddings we should be able to retrieve any word inside the analogy if given the remaining three.
There exist multiple ways to approach this problem \cite{TODO}, our main technique consists of performing additions and subtractions in order to retrieve the missing element, thus obtaining $emb(w_a) - emb(w_b) + emb(w_d) \approx emb(w_c)$.
This particular experiment allows us to elaborate on the degree of factuality and semantical properties that are stored in the embeddings of an LLM.


To measure the performance of models on this particular task we opted to show the top-$k$ accuracy for any given analogy, that is the accuracy with which the embeddings of a model are able to generate an object in the latent embedding space with the actual answer to the analogy inside its $k$ closest elements.
For this purpose, only embedded elements that correspond to an actual token of the model's vocabulary are considered when looking for the closest elements of an object.

Handling multi-token words is a vital aspect in this experiment, therefore we devised three main strategies to handle multi-token words when they are being fed to the embedding as inputs ($w_a$, $w_b$ or $w_d$) and three additional strategies to handle them as valid outputs ($w_c$) of analogies.
Strategies that work in both cases are to only consider the first token of a multi-token word (this causes results to be slightly worse on average but the impact seems to be negligible in most cases) and to reduce the dataset by only considering analogies composed of single-token words.
Other approaches applicable for multi-token words as inputs consist in taking the average or the sum over all the embedded tokens that form the multi-token word, and conversely for output multi-token words, considering all the tokens as targets for the assessment of the top-$k$ accuracy of an analogy result in the embedding space.

All the reported experiments make use of the cosine distance \cite{TODO} as a distance metric between embeddings, however every experiment has also been replicated using the L2 euclidean norm distance as a comparison outlet.
In addition, further experiments using different normalization techniques and analogy retrieval order have also been performed.


Experiments were performed using LLaMA-2-7B from Meta AI \cite{TODO} and Mistral-7B \cite{TODO} (both with a vocabulary size of 32000 and an embedding size of 4096 dimensions), GPT-2 (vocabulary size of 50257 tokens and embedding size of 768 dimensions) was also included to form a comparison against a decoder-only transformer model that is not an LLM.
It is important to note that when possible, both input embeddings and output embeddings (extrapolated from the weights of the decoder at the end of the model) were used and analyzed separately.
The dataset used to retrieve analogies is a union of the \textit{question-words} and \textit{question-phrases} datasets available through the Gensim python library \cite{TODO} and originally presented in the word2vec paper \cite{TODO} [TODO: fact check].
The complete dataset contains 19 categories of varying size and topics, for a total of 22766 analogies.
Whereas, the reduced rendition of the dataset containing only analogies composed of singe-token words was dinamically generated so that every model being analyzed was capable of encoding each word of the analogy as a single token; in particular, for the simultaneous use of the three models mentioned previously, the single-token-only dataset would contain just 2851 analogies.


By observing the results obtained on the complete dataset it's possible to notice the fact that most LLMs seem to underperform with respect to GPT-2, this striking difference in performance could be caused by multiple factors, however, by looking at the figure containing the same results on the single-token version of dataset we can observe that the performance of GPT-2 is much more in line with the other models.
This mismatch in performance may depend on the tokenization strategy and vocabulary size of the chosen models, in fact all three models use a Byte-Pair Encoding (BPE) tokenizer \cite{TODO}, however the tokenizers for both Mistral and LLaMA are based on SentencePiece \cite{TODO}, which present a smaller overall vocabulary (32000 tokens against GPT-2's 50257 tokens), possibly implying a smaller selection of full-word tokens in favor of more sub-word tokens.
By analyzing results on the single-only dataset we are removing differences that are present in how tokenizers split words, effectively normalizing away instances of analogies where GPT-2 may have an edge over the LLMs due to being able to fully encode some words, and therefore having both a better representation that encompasses the semantic meanings of the word without additional additional noise caused by multiple words sharing the same sub-word token, but also not having to be subject to the token-reconstruction strategies mentioned previously as ways to deal with multi-token words.

Another observation of note is that for bigger $k$ values, output embeddings seem to consistently outperform input embeddings (contextually to the model performance), while for smaller $k$ values input embeddings seem to present greater accuracy.
This may be caused that the fact input embeddings provide better estimates for analogies results, but their vectors are laying in latent spaces that are more sparse, thus making it less likely to accidentaly include the correct vector on a wrong prediction by extending the range given by $k$.
Whereas output embeddings latent spaces are more compressed and benefit more from greater $k$ values.
Such theory is corroborated by the fact the trend that was previously identified is much less present in the case where only single-token analogies are considered, the difference between input and output embeddings at earlier $k$ values is more accentuated, and the performance of output embeddings almost never surpasses the one of input embeddings.
This may be due to the fact that overall (and in particular for input embeddings), it is less likely for the model to output a wrong result for any given single-token only analogy, therefore the margin of analogies with wrong answers according to input embeddings and correct for output embeddings due to a large $k$ is greatly reduced.

Finally, we show that input and output embedding pairs belonging to the same model tend to have the same trend, and are generally more prone to end up having similar accuracies, which is not a completely trivial result since, despite having the same dimensionality and possibly sharing some of the datasets used to train them (depending on the training setup and initialization values), they still have completely different weight matrices and carry out different tasks inside the model.


In the end, we can say that recent LLMs still present a certain amount of facts inside their embeddings, although in a noticeably less developed and more noisy way with regards to older models.


# Question 2
– QUESTION 2: How does the output decoder relate to the input
embedding, does it simply invert the input embedding or predict
the next word?

• EXPERIMENT 2A: Multiply the two matrices and measure (i) how far
is the result from the identity matrix? (ii) how frequently is the input
word in the top 1, 5, 10, or 100 output words?

• EXPERIMENT 2B: Multiply the two matrices and measure (i) how
similar is this matrix to the first-order Markov Model transition matrix?
(ii) how frequently is the most likely next-word under a first-order MM is
in the top 1, 5, 10, 100 output words?

## Text

The second reseach question has the purpose of analyzing the relationship between the output embeddings (represented by the weights of the output decoder) and the input embeddings present in the embedding layer at the start of the model.
In particular we would like to observe if there exists any particular function that is being performed by the output decoder with regards to the input embeddings, such as inversion or even an attempt at predicting the most likely next token.


The possibility of the output embeddings performing an operation similar to an inversion of the input embeddings is due to the fact that a lot of relevant models already employ the same embeddings at the input and output of the model \cite{TODO}, thus implying a perfect inverse relationship between the two matrices representing the respective weights.

The hypothesis of output embeddings carrying out a prediction of the most likely next token based on the input, sprouts from the fact that if we were to completely deconstruct a decoder-only transformer architecture we would have the residual links directly connecting the embedding layer to the decoder \cite{TODO}.
Since the task of decoder-only models is to predict the next word given the previous context \cite{TODO}, by stripping from the architecture all the transformer blocks while keeping only a direct link from the input embedding layer to the output decoder layer, we would obtain a First Order Model (FOM) at best capable of predicting the next token given the previous one; this behavior coincides with that of first-order Markov models \cite{TODO}, which utilize bigram probability estimates (that is, given words $w_i$ and $w_j$: $P(w_j|w_i) = \frac{P(w_i, w_j)}{P(w_i)}$) retrieved from a corpus of text in order to generate the next word, given the previous one.


To verify if, and to what degree these properties hold, we have set up two main experiments: one for each hypothesis over the suggested properties of input and output embeddings pairs.
For the inverse relationship we decided to compute the distance between the matrix obtained by multiplying input and output embeddings weights together and the identity matrix with the same size.
The distance between matrices is computed using the Frobenius norm, thus obtaining $d = \| (W_{e_{in}}W_{e_{out}}^T) - I_n \|_F $ where $W_{e_{in}}$ and $W_{e_{out}}$ are the weights matrices for the input and output embeddings respectively, and $n$ represents the vocabulary size of the current model being investigated.

Additionally, we wanted to see how likely was the FOM to output the same token that was feeded to it, to this end we recorded the average top-$k$ accuracy over all the tokens inside the model's vocabulary.
The top-$k$ accuracy is computed by taking the $k$ tokens corresponding to the k highest values between logits, and observing if the input token is present in those tokens.
For the Markov model approximation, we repeated the same experiments, using a first-order Markov model trained using bigrams extracted from WikiText 2 \cite{TODO} as reference, instead of the identities used previously.
Firstly, we computed the distance from the matrix obtained as a product of the FOM embedding matrices to the transition matrix of the Markov model as done previously, then we calculated the top-$k$ accuracy for the FOM predictions by looking at the $k_1$ most likely output tokens according to the FOM and comparing them to the $k_2$ most likely output tokens for the Markov model for any given input token.


Experiments were performed on LLaMA 2 7B \cite{TODO} and Mistral 7B \cite{TODO} models, since they both present different input and output embeddings.
For each model only the input and output embeddings weights were extracted in order to create the FOM, while the rest of the model infrastructure was discarded.
The dataset used to train the Markov model by extracting bigrams is the entirety of all the splits of Wikitext 2, for a total of 44836 rows.
Whereas, for the model evaluation each single token of each model's vocabulary was used to assess the model's performance, therefore a total of 32000 tokens for both Mistral and LLaMA.


By observing the figures that represent Mistral's performance on self-regression task and bigram-regression task we can notice how, for any given $k$, the FOM's predictions seem to be closer to the inputs of the FOM rather than the most likely outputs generated by the bigram Markov model, even when considering the fact that, for the sake of this comparison, all traces (besides the one having $k_2 = 1$) are just more lenient alternatives of the respective top-$k_1$ prediction, obtained by expanding the range of valid next tokens produced by the Markov model.
This seems to suggest that the embeddings are closer to being the inverse of one another, rather than the output representing the next token prediction of the input.
Unfortunately, this result appears not to be in line with the matrix distances recorded, since we have a distance of 178.7063 between the FOM transition matrix and the identity matrix and a distance of 76.9151 between the FOM and Markov model transition matrices.

However, these results are put into perspective if we look at LLaMA's performance on the same tasks; by looking at the figure representing its perfomance on self-regression task and bigram-regression we notice opposite results, the LLaMA FOM has greater top-$k$ accuracy when compared to the bigram Markov model rather than its own inputs.
This peculiarity is noticeable by empirically looking at the FOM predictions for some common tokens, which appear to be plausible next tokens.
When observing the recorded matrix distances, they appear to be on a completely different scale than the ones obtained using Mistral's FOM, since we have a distance of 522.3115 between the FOM transition matrix and the identity matrix and a distance of 509.2659 between the FOM and Markov model transition matrices.
Even for a model that is so clearly predisposed towards providing a prediction of the next word rather than returning the input word itself, the difference between the distances is minimal, thus we could assume that this experiment is biased towards returning lower distance estimates between the FOM transition matrix and the Markov transition matrix, possibly due smaller quantities of null values present in the latter matrix with regards to the identity matrix.


In the end we can say that it seems that the feasibility of a FOM approximation is highly dependant on the model analyzed, although overall, all analyzed First Order Models composed by the combination of their input and output embeddings seem to manifest a slight bias towards trying to approximate a bigram first order Markov model.
Extracting the embedding weights from the transition matrix of a bigram Markov model and using them to initialize the embedding layer and decoder weights of an LLM would be an interesting potential development of this idea in future work.