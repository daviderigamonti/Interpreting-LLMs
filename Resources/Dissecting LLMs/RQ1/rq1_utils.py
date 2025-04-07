import itertools
import logging
import pickle
import os

from collections import defaultdict

from matplotlib.ticker import AutoMinorLocator
from matplotlib import colormaps as clmp

import matplotlib.pyplot as plt
import numpy as np
import torch

from .constants import *


# Logging
logger = logging.getLogger(LOGGER_NAME)


# Supports encoding either a single word or a list of words given a tokenizer
def multiencode(tok, words):
    # Encode a list of words
    if isinstance(words, (list, tuple)) and not isinstance(words, str):
        return torch.cat([
            tok.encode(word, return_tensors="pt", add_special_tokens=False) for word in words
        ], dim=-1)
    # Encode a single word
    return tok.encode(words, return_tensors="pt", add_special_tokens=False)
    
# Embeds a word using a specified strategy to handle the scenario where it results in multiple tokens
def strategy_encode(emb, word, tok=None, strategy="first_only"):
    # If input is a string or a list of strings tokenize and encode them, 
    if tok is not None and (isinstance(word, str) or isinstance(word[0], str)) :
        word = emb(multiencode(tok, word))
    # Use the encoding strategy only if it is needed 
    if word.shape[1] != 1:
        # Take only the first token as the resulting embedding
        if strategy == "first_only":
            word = torch.unsqueeze(word.select(1, 0), dim=1)
        # Average all token embeddings
        elif strategy == "average":
            word = torch.unsqueeze(torch.mean(word, dim=1), dim=1)
        # Sum all token embeddings
        elif strategy == "sum":
            word = torch.unsqueeze(torch.sum(word, dim=1), dim=1)
        else:
            raise Exception(f"Wrong encoding strategy")
    return word

# Calculates the distance between two words, possibly encoding and extrapolating their embeddings with the defined strategy
# The word1 parameter represents an entire embedding matrix if multi=True, thus the distance is calculated between word2 and all words
def calc_distance(emb, word1, word2, tok=None, strategy="first_only", dist="cosine", multi=False):
    # Encode and average (if multi is True, word1 represents the embedding matrix)
    if not multi:
        word1 = strategy_encode(emb, word1, tok, strategy=strategy)
    word2 = strategy_encode(emb, word2, tok, strategy=strategy)
    # Compute distances
    if dist == "L2":
        distances = torch.norm(word1 - word2, dim=2)
    elif dist == "cosine":
        cs = torch.nn.CosineSimilarity(dim=2)
        distances = 1 - cs(word1, word2)
    else:
        raise Exception("Unknown distance")
    return distances

# Returns the top-k closest elements w.r.t. a given word element inside a specified embedding space
# If decode=True and a tokenizer is specified, the results are already decoded
def get_closest_emb(emb, word, k=1, decode=True, tok=None, strategy="first_only", dist="cosine"):
    # Compute distances from matrix
    distances = calc_distance(emb, emb.weight.data, word, tok=tok, strategy=strategy, dist=dist, multi=True)
    # Compute top k smalles indices
    topk = torch.squeeze(torch.topk(distances, k=k, largest=False).indices)
    # If one element, unsqueeze it
    if k == 1:
        topk = torch.unsqueeze(topk, dim=0)
    # Decode closest k
    if decode and tok is not None:
        topk = [tok.decode(c) for c in topk.tolist()]
    return topk

# Perform embedding arithmetic on a set of words, using a sum function to specify each word's role in the arithmetic
# The resulting embedding and its closest k decoded elements are returned
def emb_arithmetic(emb, tok, words, sum_function, k=1, dist="cosine"):
    # Perform arithmetic
    result_w = sum_function(*words)
    # Normalize results if L2 distance
    result_w = torch.nn.functional.normalize(result_w, dim=-1) if dist == "L2" else result_w
    # Get closest k elements
    closest = get_closest_emb(emb, result_w, k=k, decode=True, tok=tok, dist=dist)
    return (result_w, closest)

# Performs embedding arithmetic on a batch of queries
def batch_emb_arithmetic(
    emb, tok, queries, sum_function, ref_function, k=5, strategy="first_only", dist="cosine"
):
    # Create separate normalized embeddings if L2 norm and if embeddings aren't already normalized
    nemb = emb
    if dist == "L2" and not torch.isclose(torch.norm(emb.weight, p=2, dim=-1), torch.tensor(1).to(emb.weight.dtype).to(emb.weight.device)).all():
        nemb = torch.nn.Embedding.from_pretrained(torch.nn.functional.normalize(emb.weight, dim=-1), freeze=True)
    # Compute embedding arithmetic for queries
    ret = [
        emb_arithmetic(
            nemb, tok,
            [strategy_encode(emb, word, tok, strategy=strategy) for word in q],
            sum_function,
            k=k, dist=dist
        )[1] for q in queries ]
    # Compute reference arithmetic for queries
    ref = [
        get_closest_emb(
            nemb,
            strategy_encode(nemb, ref_function(*q), strategy=strategy, tok=tok), 
            k=k, decode=True, tok=tok, strategy=strategy, dist=dist
        ) for q in queries
    ]
    return ret, ref

# Perform "delta" embedding arithmetic on a batch of queries, using a delta_el_function to specify which words should be used to compute deltas,
#  a pair_el_function/sol_el_function to specify which elements should be respectively added to the delta and which one are the solution
def batch_emb_delta_arithmetic(
    emb, tok, queries, delta_el_function, pair_el_function, sol_el_function,
    k=5, strategy="first_only", dist="cosine"
):

    # Create separate normalized embeddings if L2 norm and if embeddings aren't already normalized
    nemb = emb
    if dist == "L2" and not torch.isclose(torch.norm(emb, p=2, dim=-1), torch.tensor(1).to(emb.device)):
        nemb = torch.nn.Embedding.from_pretrained(torch.nn.functional.normalize(emb.weight, dim=-1), freeze=True)

    ret = []
    sol = []
    ref = []
    ref_pairs = defaultdict(lambda: [], {})
    # Compute reference pairs for deltas
    for q in queries:
        ref_q = q
        if not isinstance(q[0], str):
            ref_q = [qq[0] for qq in q]
        # Extract mini-batch categories to group delta computation
        category = ref_q[-1]
        # Apply delta function to select elements
        ref_pairs[category].extend([delta_el_function(*ref_q)])
    # Filter duplicate reference pairs by using sets
    ref_pairs = { category: set(frozenset(t) for t in pairs) for category, pairs in ref_pairs.items() }
    # Restore pairs with equal elements
    ref_pairs = { category: set(tuple(t) if len(t) > 1 else tuple(t) * 2 for t in pairs) for category, pairs in ref_pairs.items() }
    # Obtain the delta for each category as an average of the deltas of every category, each one obtained by subtracting the elements of
    #  reference pairs via standard embedding arithmetics; emb here doesn't matter since the result of the closest embedding is disregarded
    deltas = {
        category: torch.mean(torch.stack([
            emb_arithmetic(
                emb, tok, 
                [strategy_encode(emb, word, tok, strategy=strategy) for word in pair], 
                lambda x, y: x - y,
                k=1, dist=dist
            )[0]
            for pair in pairs
        ]).squeeze(), dim=0)
        for category, pairs in ref_pairs.items()
    }
    # Compute delta arithmetics
    for category, pairs in ref_pairs.items():
        for pair in pairs:
            # Compute results by summing each category pair reference element to the category's delta
            # If L2 distance, using the normalized embedding to encode new tensors is fundamental since the delta is normalized
            ret.append(
                emb_arithmetic(
                    nemb, tok,
                    [strategy_encode(nemb, pair_el_function(*pair), tok, strategy=strategy), deltas[category]], 
                    lambda x,y: x + y,
                    k=k, dist=dist
                )[1]
            )
            sol.append(sol_el_function(*pair))
            # Compute k-closest elements to the batch reference elements for the purpose of benchmarking against a base case
            ref.append(
                get_closest_emb(
                    nemb,
                    strategy_encode(nemb, pair_el_function(*pair), tok, strategy=strategy),
                    k=k, decode=True, tok=tok, strategy=strategy, dist=dist
                )
            )
    return ret, sol, ref

# Returns the evaluation of a batch of results against a batch of expected solutions using a given scoring metric
# If out=True, results are also directly printed on standard output
def evaluate_batch(results, solutions, out=True, score="rankscore", k=None, tok=None, ev=None):
    
    def get_rank(r, s, out=0):
        try:
            return r.index(s)
        except ValueError:
            return out

    n = len(results[0])

    # Repopulate evaluation if needed
    if ev is None:
        ev = []
        for res, sol in zip(results, solutions):
            # Get rank of each solution for each result outputs
            ranks = [get_rank(res, s, out=n) for s in sol]
            # Append best rank to final evaluation list
            ev.append(min(ranks))

    # Return score
    if score == "rankscore":
        score = 1 - ( sum(ev) / (n * len(solutions)) )
    elif score == "topk":
        if not k:
            raise ValueError("Invalid k for topk")
        if isinstance(k, list):
            score = [len([i for i in ev if i < k_el]) / len(ev) for k_el in k]
        else:
            score = len([i for i in ev if i < k]) / len(ev)
    else:
        raise ValueError("Unknown score")
    # Print scores
    if out:
        print(f"{ev} -> {score}")
    return score, ev

# Returns the evaluation of a batch of results against a batch of expected solutions using a given scoring metric
# If out=True, results are also directly printed on standard output
def evaluate_batch_categories(results, solutions, categories, score="rankscore", k=None, ev_cat=None):
    
    def get_rank(r, s, out=0):
        try:
            return r.index(s)
        except ValueError:
            return out

    n = len(results[0])

    # Repopulate evaluation if needed
    if ev_cat is None:
        ev_cat = defaultdict(lambda: [], {})
        for res, sol, cat in zip(results, solutions, categories):
            # Get rank of each solution for each result outputs
            ranks = [get_rank(res, s, out=n) for s in sol]
            # Append best rank to final evaluation list
            ev_cat[cat].append(min(ranks))

    # Return score
    scores = {}
    for cat, ev in ev_cat.items():
        if score == "rankscore":
            cat_score = 1 - ( sum(ev) / (n * len(solutions)) )
        elif score == "topk":
            if not k:
                raise ValueError("Invalid k for topk")
            if isinstance(k, list):
                cat_score = [len([i for i in ev if i < k_el]) / len(ev) for k_el in k]
            else:
                cat_score = len([i for i in ev if i < k]) / len(ev)
        else:
            raise ValueError("Unknown score")
        scores[cat] = cat_score
    return scores, ev_cat

# Encodes a batch of solutions using a specified strategy to allow appropriate matching
def encode_solutions(solutions, tok, strategy="subdivide"):
    new_solutions = []
    for sol in solutions:
        new_sol = []
        for s in sol:
            # Encode single solution
            encoded_s = tok.encode(s, return_tensors="pt", add_special_tokens=False).squeeze()
            # Use the encoding strategy only if it is needed 
            if encoded_s.size():
                # Consider all subtokens as separate valid solution
                if strategy == "subdivide":
                    new_sol.extend([tok.decode(token) for token in encoded_s])
                # Only take the first subtoken as the valid solution
                elif strategy == "first_only":
                    new_sol.extend([tok.decode(encoded_s[0])])
                else:
                    raise ValueError("Unknown solution strategy")
            else:
                new_sol.append(s)
        new_solutions.append(new_sol)
    return new_solutions

def print_results(res):
    for i, r in enumerate(res):
        print(f"{i+1}) {repr(r)}")


def change_words(batch, transform=lambda x: x):
    return [[transform(word) for word in entry] for entry in batch]

def generate_analogies_model_solutions(model_info, tokenizers, emb, df, params):
    model_id = emb["id"]
    tok = tokenizers[model_id]
    model_format = model_info[model_id]["format"]
    test_questions, test_references = batch_emb_arithmetic(
        emb["emb"], tok, 
        change_words(df[DATASET_COLUMN_FORMAT].values.tolist(), model_format), 
        params["layout"]["function"], params["layout"]["ref"], k=params["k"], 
        strategy=params["embedding_strategy"],
        dist=params["distance"],
    )
    test_question_sol = [ADDALL(entry) for entry in df[params["layout"]["solution"]].to_list()]
    return {"test_questions": test_questions, "test_question_sol": test_question_sol, "test_references": test_references}

def generate_delta_analogies_model_solutions(model_info, tokenizers, emb, df, params):
    model_id = emb["id"]
    tok = tokenizers[model_id]
    model_format = model_info[model_id]["format"]
    adv_test_questions, adv_test_question_sol, adv_test_references = batch_emb_delta_arithmetic(
        emb["emb"], tok, 
        change_words(df.values.tolist(), model_format), 
        params["layout"]["delta_function"], params["layout"]["pair_function"], params["layout"]["sol_function"],
        k=params["k"], 
        strategy=params["embedding_strategy"], dist=params["distance"],
    )
    adv_test_question_sol = [ADDALL(entry) for entry in adv_test_question_sol]
    return {"test_questions": adv_test_questions, "test_question_sol": adv_test_question_sol, "test_references": adv_test_references}

def generate_analogies_graph(model_name, emb, params, solutions, test_questions, color, **kwargs):
    rankscore, ev = evaluate_batch(test_questions, solutions,out=False, score="rankscore")
    topkscore, _ = evaluate_batch(test_questions, solutions,out=False, score='topk', k=params["test_k"], ev=ev)
    logger.debug(
        "%s %s - Rank Score: %.2f , Top-%d: %.2f", 
        model_name, emb['type'],
        rankscore,
        params["test_k"][-1],
        topkscore[-1],
    )
    plt.plot(
        params["test_k"], topkscore, 
        marker='o', alpha=0.9, label=f'{model_name} {emb['type']} Embeddings', c=color
    )

def generate_ref_analogies_graph(model_name, emb, tok, params, solutions, test_questions, test_references, color, **kwargs):
    rankscore, ev = evaluate_batch(test_questions, solutions,out=False, score="rankscore")
    topkscore, _ = evaluate_batch(test_questions, solutions,out=False, score='topk', k=params["test_k"], ev=ev)
    topkscore_ref, _ = evaluate_batch(test_references, solutions, out=False, score="topk", k=params["test_k"], tok=tok)
    logger.debug(
        "%s %s - Rank Score: %.2f , Top-%d: %.2f", 
        model_name, emb['type'],
        rankscore,
        params["test_k"][-1],
        topkscore[-1],
    )
    label = kwargs["label"] if "label" in kwargs else f"{model_name} {emb['type']}"
    plt.plot(
        params["test_k"], topkscore, 
        marker='o', alpha=0.9, label=f'{label} Embeddings', c=color
    )
    if "baselines" not in kwargs or kwargs["baselines"]:
        plt.plot(
            params["test_k"], topkscore_ref, 
            alpha=0.5, linestyle = '--', c=color
        )

def create_analogy_models_folder(model_info, embeddings):
    return f"{'_'.join([(model_info[emb['id']]['name'] + '.' + emb['type']).replace(' ', '') for emb in embeddings])}"

def create_analogy_dump_name(params, emb, format_params):
    return f"{emb['id'].split("/")[-1]}.{emb['type']}_{params['distance']}-emb_{params['embedding_strategy']}-{params['layout']['vis'].replace(' ', '')}{'-prenormalize' if params['pre_normalize'] else ''}.pkl"

def create_analogy_image_name(params, format_params):
    return f"{params['distance']}-emb_{params['embedding_strategy']}-multisol_{params['multitoken_solutions_strategy']}\
-{params['layout']['vis'].replace(' ', '')}{'-prenormalize' if params['pre_normalize'] else ''}.{format_params['fig_ext']}"

def create_analogy_title_name(params):
    return f"Gensim sum analogies {params['layout']['vis']} ({params['distance']} distance) (embeddings: {params['embedding_strategy']}) (multitoken solutions: {params['multitoken_solutions_strategy']})\
 (dataset: {'only single tokens' if FILTER_SINGLE_TOKENS else 'complete'}) (pre normalization: {str(params['pre_normalize'])})"

def create_log_title_name(model_info, embeddings, params):
    logger.debug(
        "Executing tests for:\n- Models: %s\n- Parameters: \n  -%s", 
        ", ".join([f"{mod['name']}({'/'.join([emb['type'] for emb in embeddings if emb['id'] == k])})" for k, mod in model_info.items()]),
        "\n  -".join([f"{k}({'/'.join([str(el) if not isinstance(el, dict) else el['vis'] for el in v])})" for k,v in params.items()])
    )

def batch_graph_generation(model_info, tokenizers, embeddings, df, analogies_type, kv_loop_params, format_params):

    create_log_title_name(model_info, embeddings, kv_loop_params)
    
    new_folder_path = format_params["folder_path"] + create_analogy_models_folder(model_info, embeddings) + "/"
    format_params = format_params | {
        "folder_path": new_folder_path,
        "dumps_path": new_folder_path + "dumps/",
    }
    # Make sure directory exists
    if not os.path.exists(format_params["folder_path"]):
        os.makedirs(format_params["folder_path"])
    if not os.path.exists(format_params["dumps_path"]):
        os.makedirs(format_params["dumps_path"])

    # Create parameter grid
    loop_k, loop_v = zip(*kv_loop_params.items())
    loop = [dict(zip(loop_k, p)) for p in itertools.product(*loop_v)]

    for p in loop:
        params = dict(p)

        image_name = create_analogy_image_name(params, format_params)
        full_title = create_analogy_title_name(params)

        image_name = format_params["folder_path"] + image_name
        logger.debug(full_title)
        
        if format_params["skip"] and os.path.exists(image_name):
            logger.debug("Skipping")
            continue

        plt.figure()

        for (model_id, model), trace_idx in zip(model_info.items(), range(0, len(model_info)*2, 2)):
            model_name = model["name"]
            tok = tokenizers[model_id]
            indexed_model_embeddings = [e | {"idx": idx} for idx, e in enumerate(embeddings) if e["id"] == model_id]

            for i, emb in enumerate(indexed_model_embeddings):

                torch.cuda.empty_cache()

                dev_emb = emb | {
                    "emb": (
                        torch.nn.Embedding.from_pretrained(torch.nn.functional.normalize(emb["emb"].weight, dim=-1), freeze=True) if params["pre_normalize"] else emb["emb"]
                    ).to(DEVICE) 
                }

                # Compute embedding analogy solutions
                dump_name = format_params["dumps_path"] + create_analogy_dump_name(params, emb, format_params)
                if os.path.exists(dump_name):
                    with open(dump_name, "rb") as f:
                        test_sol_info = pickle.load(f)
                else:
                    test_sol_info = ANALOGY_SOL_MAP[analogies_type](model_info, tokenizers, dev_emb, df, params)
                    with open(dump_name, "w+b") as f:
                        pickle.dump(test_sol_info, f)

                # ONLY FOR VISUALIZATION
                # Encode and format solutions
                #formatted_solution = [[model["format"](e) for e in el] for el in test_sol_info["test_question_sol"]]
                #solutions = encode_solutions(formatted_solution, tok, strategy=params["multitoken_solutions_strategy"])

                # Add embedding to graph
                #ANALOGY_GRAPH_MAP[analogies_type](trace_idx + i, model_name, dev_emb, tok, params, solutions, **test_sol_info)
            
            # ONLY FOR VISUALIZATION
            #plt.xlabel('k')
            #plt.ylabel('Top-k Accuracy')
            #plt.xticks(params["test_k"])

        # ONLY FOR VISUALIZATION
        #plt.title(full_title, fontsize=12)
        #plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        #plt.gca().xaxis.set_minor_locator(AutoMinorLocator(1))
        #plt.grid(linestyle = '--', linewidth = 0.5, which="minor")
        #plt.grid(linestyle = '--', linewidth = 1, which="major")
        if format_params["save_fig"]:
            plt.savefig(image_name, bbox_inches='tight')
        if format_params["show_fig"]:
            plt.show()
        plt.close()

def batch_graph_visualization(model_info, tokenizers, embeddings, df, analogies_type, kv_loop_params, format_params):

    create_log_title_name(model_info, embeddings, kv_loop_params)
    
    new_folder_path = format_params["folder_path"] + create_analogy_models_folder(model_info, embeddings) + "/"
    format_params = format_params | {
        "folder_path": new_folder_path,
        "dumps_path": new_folder_path + "dumps/",
    }
    # Make sure directory exists
    if not os.path.exists(format_params["folder_path"]):
        os.makedirs(format_params["folder_path"])
    if not os.path.exists(format_params["dumps_path"]):
        os.makedirs(format_params["dumps_path"])

    # Create parameter grid
    loop_k, loop_v = zip(*kv_loop_params.items())
    loop = [dict(zip(loop_k, p)) for p in itertools.product(*loop_v)]

    categories = df["category"].tolist()
    n_cat = len(df["category"].unique().tolist())

    for p in loop:
        params = dict(p)

        image_name = "vis_" + create_analogy_image_name(params, format_params)
        full_title = create_analogy_title_name(params)

        image_name = format_params["folder_path"] + image_name
        logger.debug(full_title)

        plt.figure(figsize=(25,12))
        s = 0.8
        n_bar = len(embeddings)
        w = s / n_bar

        for idx, ((model_id, model), trace_idx) in enumerate(zip(model_info.items(), range(0, len(model_info)*2, 2))):
            model_name = model["name"]
            tok = tokenizers[model_id]
            indexed_model_embeddings = [e | {"idx": idx} for idx, e in enumerate(embeddings) if e["id"] == model_id]

            for i, emb in enumerate(indexed_model_embeddings):

                torch.cuda.empty_cache()

                dev_emb = emb | {
                    "emb": (
                        torch.nn.Embedding.from_pretrained(torch.nn.functional.normalize(emb["emb"].weight, dim=-1), freeze=True) if params["pre_normalize"] else emb["emb"]
                    ).to(DEVICE) 
                }

                # Compute embedding analogy solutions
                dump_name = format_params["dumps_path"] + create_analogy_dump_name(params, emb, format_params)
                if os.path.exists(dump_name):
                    with open(dump_name, "rb") as f:
                        test_sol_info = pickle.load(f)
                else:
                    continue

                # Encode and format solutions
                formatted_solution = [[model["format"](e) for e in el] for el in test_sol_info["test_question_sol"]]
                solutions = encode_solutions(formatted_solution, tok, strategy=params["multitoken_solutions_strategy"])

                topkscore, _ = evaluate_batch_categories(test_sol_info["test_questions"], solutions, categories, score='topk', k=params["test_k"])
                plt.bar(
                    [j + (((idx + i) / n_bar) * s) - (((w * (n_bar - 1)) / 2)) for j in range(len(topkscore))],
                    [l[-1] for l in topkscore.values()],
                    width = w
                )
                
            plt.xlabel('categories')
            plt.ylabel('Top-k Accuracy')
            plt.xticks(
                np.arange(len(topkscore)), #np.arange(0, len(topkscore), 1 / (len(indexed_model_embeddings) + 1)),
                list(topkscore.keys()),
                rotation=40, ha="right", va="top"
            )

        plt.title(full_title, fontsize=12)
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.xlim(-1, n_cat + 1)
        plt.ylim(0, 1.05)
        plt.grid(axis="y", linestyle = '--', linewidth = 0.5, which="minor")
        plt.grid(axis="y", linestyle = '--', linewidth = 1, which="major")
        
        if format_params["save_fig"]:
            plt.savefig(image_name, bbox_inches='tight')
        if format_params["show_fig"]:
            plt.show()
        plt.close()

ANALOGY_SOL_MAP = {
    "analogies": generate_analogies_model_solutions,
    "delta_analogies": generate_delta_analogies_model_solutions,
}

ANALOGY_GRAPH_MAP = {
    "analogies": generate_ref_analogies_graph,
    "delta_analogies": generate_ref_analogies_graph,
}
