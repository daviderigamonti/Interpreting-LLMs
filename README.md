# Interpreting Large Language Models Through the Lens of Embedding-Oriented Visualizations: Markov Models, Sankey Diagrams and Comparative Approaches

In recent years, the progress and widespread usage of Large Language Models (LLMs) has steadily increased.
Understanding the internal mechanisms of these complex and opaque engineering products poses an extremely relevant and critical challenge in the field of Natural Language Processing (NLP).
The NLP interpretability subfield aims to provide insights into the inner workings of language models (LMs).
This thesis presents InTraVisTo, an interactive visualization tool designed to provide an immediate and intuitive perspective on the generation process of LLMs by decoding their internal states, normally incomprehensible by humans, using tokens from their vocabularies.
InTraVisTo is implemented as an interactive framework that enables tracking of the flow of information through residual connections between model components, and it allows users to directly sway the model's predictions by modifying internal representations.
Additionally, this research investigates whether recent LLMs retain linear properties in their embedding spaces through the established semantic analysis framework of word analogies, finding that current state-of-the art LLMs are capable of encoding a surprising amount of semantic relationships within their embeddings.
Furthermore, this work examines the feasibility of obtaining first-order predictions by concatenating input and output embeddings in large-scale models, testing whether LLMs inherently approximate a Markov model structure.
Our findings demonstrate that first-order models (FOMs) extracted from LLMs exhibit a clear bias towards modeling a Markovian behavior, despite significant differences in accuracy across model architectures.
Overall, this thesis confirms previous findings on new architectures and introduces a robust tool aimed at making LLMs more transparent and accessible to researchers in the NLP field.

## Abstract in Italiano

Negli ultimi anni, lo sviluppo e la diffusione su larga scala dei Large Language Model (LLM) hanno registrato un costante incremento.
Comprendere i meccanismi interni di simili prodotti ingegneristici di natura complessa e opaca costituisce una sfida di rilevanza e importanza critica nel campo dell'Elaborazione del Linguaggio Naturale (NLP); di conseguenza il sottocampo dell'interpretabilità nell'ambito dell'NLP si propone di fornire una comprensione approfondita del funzionamento interno dei modelli linguistici (LM).
In questa tesi si presenta InTraVisTo, uno strumento di visualizzazione interattiva concepito per offrire una prospettiva intuitiva ed immediata sul processo generativo svolto dagli LLM mediante la decodifica dei loro stati interni, altrimenti umanamente incomprensibili, attraverso l'impiego dei token contenuti nei loro vocabolari.
InTraVisTo viene implementato in qualità di framework interattivo, capace di fornire visualizzazioni adatte a monitorare il flusso informativo osservando le connessioni residue tra i vari componenti, e incentrato sul permettere agli utenti di intervenire influenzando direttamente le previsioni del modello attraverso la modifica delle sue rappresentazioni interne.
Inoltre, questa ricerca prende in considerazione gli LLM più recenti al fine di verificare se conservino le proprietà lineari all'interno dei rispettivi spazi di embedding, adoperando la consolidata metodologia di analisi semantica basata sulle analogie tra parole, ed evidenziando come i modelli di ultima generazione siano capaci di codificare un sorprendente numero di relazioni semantiche nei loro embedding.
In aggiunta, questo lavoro tratta la possibilità di ottenere previsioni di primo ordine attraverso la concatenazione degli embedding di input e output su modelli di grandi dimensioni, verificando se gli LLM siano capaci di approssimare intrinsecamente la struttura di un modello di Markov.
I risultati ottenuti evidenziano che i modelli di primo ordine (FOM) derivati dagli LLM manifestano una chiara propensione verso un comportamento Markoviano, sebbene emergano significative differenze di accuratezza in funzione delle architetture adottate.
In sintesi, la presente tesi riconferma risultati storici su nuove architetture e introduce un robusto strumento finalizzato a rendere gli LLM maggiormente trasparenti e accessibili alla comunità di ricerca nel campo dell'NLP.

## Errata Corrige

### Thesis

#### Text

- **Section 4.3.2 Paragraph 1:** As a preliminary [...] by combining ~~and transposing~~ the input and output [...]. &rarr; As a preliminary [...] by combining the input and output [...].
- **Section 4.3.4 Paragraph 3:** Where $S$ denotes a sentence composed of a total of $\cancel{T}$ tokens identified as $\cancel{S(1),\dots,S(T)}$, [...]. &rarr; Where $S$ denotes a sentence composed of a total of $T+1$ tokens identified as $S(0),\dots,S(T)$, [...].
- **Section 6 Paragraph 4:** Additional developments addressing [...] on the embedding ~~space LLMs~~, as well as [...]. &rarr; Additional developments addressing [...] on the embedding space of LLMs, as well as [...].
- **Appendix A:** [...] repository containing the $\LaTeX\!$ ~~source~~ code [...]. &rarr; [...] repository containing the $\LaTeX$ source code [...].

#### Formulas

- **Formula 4.22:** $`\mathbf{Q}_{\textit{FOM}} = \text{softmax}(\mathbf{Q}_{\textit{FOM}}^{log}) = \text{softmax}(\mathbf{W}_{in} \cdot \mathbf{W}_{out}^\mathrm{T})`$ &rarr; $`\mathbf{Q}_{\textit{FOM}} = \text{softmax}(\mathbf{Q}_{\textit{FOM}}^{log}) = \text{softmax}(\mathbf{W}_{in} \cdot \mathbf{W}_{out})`$
- **Formula 4.23 (1):** $`\mathbf{Q}_{\textit{FOM}}^{RMS} = \text{softmax}(\mathbf{W}_{in,RMS} \cdot \mathbf{W}_{out,RMS}^\mathrm{T})`$ &rarr; $`\mathbf{Q}_{\textit{FOM}}^{RMS} = \text{softmax}(\mathbf{W}_{in,RMS} \cdot \mathbf{W}_{out,RMS})`$
- **Formula 4.29:** $`PP(\mathbf{Q}, S) = e^{-\frac{1}{T}\sum_{t=1}^{T}{\ln{\left((\mathbf{Q})_{S(t+1),S(t)}\right)}}}`$ &rarr; $`PP(\mathbf{Q}, S) = e^{-\frac{1}{T}\sum_{t=1}^{T}{\ln{\left((\mathbf{Q})_{S(t-1),S(t)}\right)}}}`$
- **Formula 4.30 (2):** $`D_{KL}^v(\mathbf{Q}_{ref} || \mathbf{Q}_{model}) = \sum_{w \in \mathcal{V}}{\bigl((\mathbf{Q}_{ref})_{v,w}\bigr)\ln{\frac{(\mathbf{Q}_{ref})_{v,w}}{(\mathbf{Q}_{model})_{v,\cdot}}}}`$ &rarr; $`D_{KL}^v(\mathbf{Q}_{ref} || \mathbf{Q}_{model}) = \sum_{w \in \mathcal{V}}{\bigl((\mathbf{Q}_{ref})_{v,w}\bigr)\ln{\frac{(\mathbf{Q}_{ref})_{v,w}}{(\mathbf{Q}_{model})_{v,w}}}}`$

## Info

- **Author:** Davide Rigamonti
- **University:** Politecnico di Milano
- **Course:** Computer Science and Engineering / Ingegneria Informatica
- **Academic Year:** 2023-24
- **Advisor:** Prof. Mark Carman
- **Co-advisors:** Nicolò Brunello, Vincenzo Scotti
