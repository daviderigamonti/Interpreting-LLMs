from torcheval.metrics.functional.text import perplexity
from matplotlib.ticker import FixedLocator, AutoMinorLocator
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch

# Matrix distance metric between two matrices
def matrix_distance(m1, m2):
    # return torch.linalg.svdvals(m1.cpu()-m2.cpu()).max()
    return torch.norm((m1 - m2).to("cuda"), p="fro")

# Compute distance between a matrix and the corresponding identity matrix
def distance_from_I(matrix):
    n = matrix.size(0) # Use dimension 0 to build identity matrix
    I = torch.eye(n)
    return matrix_distance(matrix, I)

# Compute the most probable k words for each element inside the vocabulary of a given transition matrix
def next_words(trans_matrix, k=1):
    n = trans_matrix.size(0)
    logits_results = []
    topk_results = []
    for i in range(0, n):
        # Compute logits and probabilities
        logits = trans_matrix[i, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        logits_results.append(logits)
        # Compute top-k indices corresponding to most probable words
        topk = torch.squeeze(torch.topk(logits, k=k, largest=True).indices).detach().cpu().int().numpy().tolist()
        topk_results.append(topk)
    return logits_results, topk_results

def vis_topk_result(tokenizer, topk, i):
    return f"{tokenizer.decode([i])} -> {', '.join([tokenizer.decode(top) for top in topk[i]])}"

# Computes the top-k accuracy for the predictions of a model described by its transition matrix against a reference
def compute_topk_accuracy(trans_matrix, reference, k=1):
    # Handle a possible list of ks
    if type(k) != list:
        k = [k]

    # Compute actual top-k predictions
    # (if the k parameter is a list, the maximum k is used to avoid retrieving predictions multiple times)
    logits, topk = next_words(trans_matrix, k=max(k))
    
    # Handle lists with k=1
    reference = [[el] for el in reference] if not isinstance(reference[0], list) else reference
    topk = [[el] for el in topk] if not isinstance(topk[0], list) else topk

    results = []
    # Compute accuracy for all specified k values
    for current_k in k:
        hits = sum((bool(set(result[:current_k]).intersection(set(ref))) for result, ref in zip(topk, reference)))
        results.append(hits / len(reference))
    return results if len(results) > 1 else results[0]


def fom_comparison_graph(trans_matrix1, ks_1, trans_matrix2=None, ks_2=[], image_name=None):
    if trans_matrix2 is not None:
        for k_2 in ks_2:  
            _, reference = next_words(trans_matrix2, k=k_2)
            topk_results = compute_topk_accuracy(trans_matrix1, reference, k=ks_1)
            plt.plot(ks_1, topk_results, marker='o', label=f'Top-{k_2} ($k_2$) Accuracy')
            plt.xlabel('$k_1$')
    else:
        reference = list(range(0, trans_matrix1.size(0)))
        topk_results = compute_topk_accuracy(trans_matrix1, reference, k=ks_1)
        plt.plot(ks_1, topk_results, marker='o', label="Top-$k$ Accuracy")
        plt.xlabel('$k$')

    plt.ylabel('Accuracy')
    # Title
    # plt.title(f"Top-k accuracy for LLaMA FOM / Markov", fontsize=12)

    # plt.gca().xaxis.set_minor_locator(AutoMinorLocator(1)) # Print as PNG
    plt.gca().xaxis.set_minor_locator(FixedLocator([15, 30, 40, 66.67, 83.33])) # Print as PDF
    plt.gca().xaxis.set_major_locator(FixedLocator(ks_1)) # Print as PDF

    plt.grid(linestyle = '--', linewidth = 0.5, which="minor")
    plt.grid(linestyle = '--', linewidth = 1, which="major")
    plt.legend(loc="lower right", prop={'size': 9})
    if image_name:
        # plt.savefig(f"Q2_{model_name}_FOM_topk_Markov.png", bbox_inches='tight') # Print as PNG
        plt.savefig(image_name, bbox_inches='tight')
    plt.close()

# Create the transition matrix for a 1-gram Markov model, given token/frequency dataframe
def build_transition_matrix(df, emb, smooth=0):
    emb_matrix = emb.weight.data
    n = emb_matrix.size(0)
    trans_matrix = torch.full((n, n), fill_value=smooth)
    for idx, row in df.iterrows():
        trans_matrix[row["Token1"], row["Token2"]] += row["Count"]
    # Normalize matrix
    trans_matrix = trans_matrix / trans_matrix.sum(dim=1, keepdim=True)
    trans_matrix[trans_matrix != trans_matrix] = 0 # Remove nans obtained by possible divisions by 0
    return trans_matrix


def overlap_coef(set1, set2):
    return len(set1.intersection(set2)) / min((len(set1), len(set2)))

def jaccard_sim(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))

def sorensen_sim(set1, set2):
    return 2 * len(set1.intersection(set2)) / (len(set1) + len(set2))

def test_set_sim(set1, set2, baseline_set, n_k, sim_f):
    list_overlap = []
    list_overlap_b1 = []
    list_overlap_b2 = []
    for k in n_k:
        _set1 = [set(s[:k]) for s in set1]
        _set2 = [set(s[:k]) for s in set2]
        _baseline_set = [set(b[:k]) for b in baseline_set]

        overlap = [sim_f(s1, s2) for s1,s2 in zip(_set1, _set2)]
        overlap_b1 = [sim_f(s1, b) for s1,b in zip(_set1, _baseline_set)]
        overlap_b2 = [sim_f(s2, b) for s2,b in zip(_set2, _baseline_set)]

        list_overlap.append(sum(overlap) / len(overlap))
        list_overlap_b1.append(sum(overlap_b1) / len(overlap_b1))
        list_overlap_b2.append(sum(overlap_b2) / len(overlap_b2))

    return list_overlap, list_overlap_b1, list_overlap_b2

def graph_test_sim(
    metrics, n_k, limit=None, 
    xlabel="", ylabel="", title="",
    plotxlim=None, plotylim=None,
    noxlabels=False,
    image_name=None
):
    for metric in metrics:
        plt.plot(n_k[:limit], metric["values"][:limit], **metric["style"])

    plt.title(title, fontsize=12)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(1))

    plt.grid(linestyle = '--', linewidth = 0.5, which="minor")
    plt.grid(linestyle = '--', linewidth = 1, which="major")

    legend = plt.legend(prop={'size': 9})
    # Avoid transparency in legend
    for lh in legend.legend_handles: 
        lh.set_alpha(1)

    if plotxlim:
        plt.xlim(*plotxlim)
    if plotylim:
        plt.ylim(*plotylim)

    if noxlabels:
        plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    if image_name:
        plt.savefig(image_name, bbox_inches='tight')

    plt.close()

def compute_perplexity(dataset, tokenizer, predict):
    perp = []
    for txt in tqdm(dataset):
        perp_logits = []
        perp_truths = []
        prev_token = -1
        if len(tokenizer.encode(txt["text"])) > 1:
            for cur_token in tokenizer.encode(txt["text"]):
                if prev_token != -1:
                    logits = predict(prev_token)
                    perp_logits.append(logits)
                    perp_truths.append(cur_token)
                prev_token = cur_token
            perp.append(perplexity(torch.stack(perp_logits).unsqueeze(0), torch.IntTensor(perp_truths).unsqueeze(0)).cpu())
    return torch.stack(perp)
