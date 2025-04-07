import sys
sys.path.append("/home/daviderigamonti/Thesis/dissecting_llm")

import os
os.chdir("/home/daviderigamonti/Thesis")

import logging

from src.RQ1_2.load_utils import *
from src.RQ1_2.RQ1.data_utils import *
from src.RQ1_2.RQ1.rq1_utils import *

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-nm", dest="numbermodel")

args, unknown = parser.parse_known_args()

MODEL_INFO_REF = {
    "word2vec": {"name": "Word2Vec", "format": NOADD, "color": "tab:brown"},
    "glove": {"name": "GloVe", "format": NOADD, "color": "tab:red"},
    "google-bert/bert-large-uncased": {"name": "BERT", "format": NOADD, "color": "tab:green"},
    "gpt2": {"name": "GPT 2", "format": ADDSPACE, "color": "tab:olive"},
    "google/gemma-2-2b": {"name": "GEMMA", "format": ADDSPACE, "color": "tab:orange"},
    "meta-llama/Llama-2-7b-hf": {"name": "LLaMa 2", "format": NOADD, "color": "tab:blue"},
    "meta-llama/Meta-Llama-3-8B": {"name": "LLaMa 3", "format": NOADD, "color": "tab:cyan"},
    "mistralai/Mistral-7B-v0.3": {"name": "Mistral v3", "format": NOADD, "color": "tab:purple"},
    "microsoft/Phi-3.5-mini-instruct": {"name": "Phi 3", "format": NOADD, "color": "tab:pink"}
}
MODEL_INFO = {args.numbermodel: MODEL_INFO_REF[args.numbermodel]}

# Formatting functions, used to change the format of inputs to fit models' needs
NOADD = lambda x: x
ADDSPACE = lambda x: " " + x if x[0] != " " else x
ADDALL = lambda x: (x.title(), x.lower())

# Define models
#MODEL_INFO = {
    #"word2vec": {"name": "Word2Vec", "format": NOADD, "color": "tab:brown"},
    #"glove": {"name": "GloVe", "format": NOADD, "color": "tab:red"},
    #"google-bert/bert-large-uncased": {"name": "BERT", "format": NOADD, "color": "tab:green"},
    #"gpt2": {"name": "GPT 2", "format": ADDSPACE, "color": "tab:olive"},
    #"google/gemma-2-2b": {"name": "GEMMA", "format": ADDSPACE, "color": "tab:orange"},
    #"meta-llama/Llama-2-7b-hf": {"name": "LLaMa 2", "format": NOADD, "color": "tab:blue"},
    #"meta-llama/Meta-Llama-3-8B": {"name": "LLaMa 3", "format": NOADD, "color": "tab:cyan"},
    #"mistralai/Mistral-7B-v0.3": {"name": "Mistral v3", "format": NOADD, "color": "tab:purple"},
    #"microsoft/Phi-3.5-mini-instruct": {"name": "Phi 3", "format": NOADD, "color": "tab:pink"}
#}
DEVICE = "cuda"
NEED_KEY = [ "meta-llama/Llama-2-7b-hf", "google/gemma-7b", "meta-llama/Llama-2-13b-hf" ]
INCLUDE_OUTPUT_EMB = True
LINEAR_INTERPOLATION = []

# Path of dataset files
DATASET_PATHS = ["dissecting_llm/src/data/analogy/questions-words.txt", "dissecting_llm/src/data/analogy/questions-phrases.txt", "dissecting_llm/src/data/analogy/bats/bats.txt"]
# Dataset column format for referencing
DATASET_COLUMN_FORMAT = ["e1", "e2", "e3", "e4"]
# Subset of categories to analyze, if empty, all categories are loaded
SELECT_CATEGORIES = []
# If True, remove all analogies that are not entirely composed by words that can be encoded using single tokens by all loaded models
FILTER_SINGLE_TOKENS = False
# BATS import constants
BATS_DIRECTORY = "dissecting_llm/src/data/analogy/bats"
NEW_BATS_FILE = "dissecting_llm/src/data/analogy/bats/bats.txt"
SKIP_CODES = []
CAPITALIZE_CODES = ["E01", "E02", "E03", "E04"]
MAX_SYNONYMS = 3

# Execution parameters
PARAMETERS = {
    "test_k": [[5, 10, 15, 25, 50]],
    "k": [50],
    "distance": ["cosine"], #["cosine", "L2"],
    "embedding_strategy": ["first_only", "average", "sum"] if not FILTER_SINGLE_TOKENS else ["first_only"],
    "multitoken_solutions_strategy": ["subdivide"], #["first_only", "subdivide"],
    "pre_normalize": [False, True],
    "layout": [
        {"function": lambda e1, e2, e3, e4: e1 - e2 + e4, "ref": lambda e1, e2, e3, e4: e4, "solution": "e3", "vis": "e1 - e2 + e4 = e3"},
        {"function": lambda e1, e2, e3, e4: e2 - e1 + e3, "ref": lambda e1, e2, e3, e4: e3, "solution": "e4", "vis": "e2 - e1 + e3 = e4"},
    ],
}
PARAMETERS_DELTA = {
    "test_k": [[5, 10, 15, 25, 50]],
    "k": [50],
    "distance": ["cosine"], #["cosine", "L2"],
    "embedding_strategy": ["first_only", "average", "sum"] if not FILTER_SINGLE_TOKENS else ["first_only"],
    "multitoken_solutions_strategy": ["subdivide"], #["first_only", "subdivide"],
    "pre_normalize": [False, True],
    "layout": [
        {"delta_function": lambda e1, e2, e3, e4, *args: (e4, e3), "pair_function": lambda e4, e3: e3, "sol_function": lambda e4, e3: e4, "vis": "e1 + Δ(e4 - e3) = e2"}
    ],
}

FORMAT_PARAMETERS = {
    "skip": True,
    "folder_path": "dissecting_llm/src/RQ1_2/RQ1/img/vis_analogies/" + ("single_tokens" if FILTER_SINGLE_TOKENS else "complete") + "/",
    "ref_path": "dissecting_llm/src/RQ1_2/RQ1/img/analogies/" + ("single_tokens" if FILTER_SINGLE_TOKENS else "complete") + "/",
    "show_fig": False,
    "save_fig": True,
    "fig_ext": "png",
}
FORMAT_PARAMETERS_DELTA = {
    "skip": True,
    "folder_path": "dissecting_llm/src/RQ1_2/RQ1/img/delta-vis_analogies/" + ("single_tokens" if FILTER_SINGLE_TOKENS else "complete") + "/",
    "ref_path": "dissecting_llm/src/RQ1_2/RQ1/img/delta-analogies/" + ("single_tokens" if FILTER_SINGLE_TOKENS else "complete") + "/",
    "show_fig": False,
    "save_fig": True,
    "fig_ext": "png",
}

# Logging
LOG_PATH = "dissecting_llm/src/RQ1_2/RQ1/logs/"
LOGGER_NAME = "rq1_logger"

# Logging
setup_logger(LOG_PATH, LOGGER_NAME, "vis_analogies", logging.DEBUG)
logger = logging.getLogger(LOGGER_NAME)

# Load models
torch.set_default_device("cpu")

tokenizers, embeddings = extract_embeddings(
    MODEL_INFO, NEED_KEY,
    rms_output=False, include_out=INCLUDE_OUTPUT_EMB, linear_interpolation=LINEAR_INTERPOLATION
)

# Import separate question-words datasets for each model
dfs = {
    key: import_question_words(
        DATASET_PATHS, DATASET_COLUMN_FORMAT, SELECT_CATEGORIES, FILTER_SINGLE_TOKENS, MODEL_INFO,
        tokenizers={key: tok}
    )[0] for key, tok in tokenizers.items()
}

ref_df, ref_categories = import_question_words(DATASET_PATHS, DATASET_COLUMN_FORMAT, SELECT_CATEGORIES, False)
ref_cat_n = [len(ref_df[ref_df["category"] == cat]) for cat in ref_categories]
n_cat = len(ref_categories)

support = {
    key: [len(df[df["category"] == cat]) / cat_n for cat, cat_n in zip(ref_categories, ref_cat_n)]
    for key, df in dfs.items()
}

model_info = MODEL_INFO
tokenizers =  tokenizers
embeddings = embeddings
dfs = dfs 
analogies_type = "vis_analogies"
kv_loop_params = PARAMETERS
format_params = FORMAT_PARAMETERS

create_log_title_name(model_info, embeddings, kv_loop_params)
    
new_folder_path = format_params["folder_path"] + create_analogy_models_folder(model_info, embeddings) + "/"
new_format_params = format_params | {
    "folder_path": new_folder_path,
    "ref_path": format_params["ref_path"],
}

# Make sure directory exists
if not os.path.exists(new_format_params["folder_path"]):
    os.makedirs(new_format_params["folder_path"])

# Create parameter grid
loop_k, loop_v = zip(*kv_loop_params.items())
loop = [dict(zip(loop_k, p)) for p in itertools.product(*loop_v)]

for p in loop:
    params = dict(p)

    image_name = "categorybar_" + create_analogy_image_name(params, new_format_params)
    image_name = new_format_params["folder_path"] + image_name

    full_title = create_analogy_title_name(params)
    logger.debug(full_title)

    plt.figure(figsize=(40,12))

    target_k = -2
    s = 0.8
    n_bar = len(embeddings)
    w = s / n_bar

    for idx, (model_id, model) in enumerate(model_info.items()):
        model_name = model["name"]
        tok = tokenizers[model_id]
        supp = support[model_id]
        categories = dfs[model_id]["category"].tolist()
        indexed_model_embeddings = [e | {"idx": idx} for idx, e in enumerate(embeddings) if e["id"] == model_id]

        for i, emb in enumerate(indexed_model_embeddings):

            # Compute embedding analogy solutions
            dump_name = new_format_params["ref_path"] + (model_name + '.' + emb['type']).replace(' ', '') + "/dumps/"
            dump_name = dump_name + create_analogy_dump_name(params, emb, new_format_params)
            if os.path.exists(dump_name):
                with open(dump_name, "rb") as f:
                    test_sol_info = pickle.load(f)
            else:
                # Alternative dump name 
                dump_name = new_format_params["ref_path"] + '_'.join([(model_info[emb['id']]['name'] + '.' + emb['type']).replace(' ', '') for emb in embeddings]) + "/dumps/"
                dump_name = dump_name + create_analogy_dump_name(params, emb, new_format_params)
                if os.path.exists(dump_name):
                    with open(dump_name, "rb") as f:
                        test_sol_info = pickle.load(f)
                else:
                    logger.debug("Dump not found")

            # Encode and format solutions
            formatted_solution = [[model["format"](e) for e in el] for el in test_sol_info["test_question_sol"]]
            solutions = encode_solutions(formatted_solution, tok, strategy=params["multitoken_solutions_strategy"])

            topkscore, _ = evaluate_batch_categories(test_sol_info["test_questions"], solutions, categories, score='topk', k=params["test_k"])
            
            topkscore = [topkscore[cat][target_k] if cat in topkscore else 0.0 for cat in ref_categories]
            x_positions = [j + (((idx + i) / n_bar) * s) - (((w * (n_bar - 1)) / 2)) for j in range(len(topkscore))]

            plt.bar(
                x_positions,
                [l for l in topkscore],
                width = w, align="center", zorder=2,
                color = model["color"],
                label = f"{model_name} {emb['type']} Embeddings"
            )
            if FILTER_SINGLE_TOKENS:
                plt.scatter(
                    x_positions,
                    [l for l in supp],
                    color="gray", edgecolor="black", s=25, linewidths=0.8, zorder=4,  
                    label=f"Support for {model_name}"
                )
                plt.vlines(
                    x_positions, 
                    ymin=0, ymax=[l for l in supp], 
                    color="gray", linestyles=(0, (5, 10)), linewidths=0.8, zorder=2,
                )
            
        plt.xlabel('categories')
        plt.ylabel(f'Top-{params["test_k"][target_k]} Accuracy')
        plt.xticks(
            np.arange(len(ref_categories)), 
            ref_categories,
            rotation=40, ha="right", va="top"
        )

    plt.title(full_title, fontsize=12)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.xlim(-1, n_cat + 1)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle = '--', linewidth = 0.5, which="minor")
    plt.grid(axis="y", linestyle = '--', linewidth = 1, which="major")

    if new_format_params["save_fig"]:
        plt.savefig(image_name, bbox_inches='tight')
    if new_format_params["show_fig"]:
        plt.show()
    plt.close()

for p in loop:
    params = dict(p)

    image_name = "kacc_" + create_analogy_image_name(params, new_format_params)
    image_name = new_format_params["folder_path"] + image_name
    
    full_title = create_analogy_title_name(params)
    logger.debug(full_title)

    plt.figure()

    for idx, (model_id, model) in enumerate(model_info.items()):
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
            dump_name = new_format_params["ref_path"] + (model_name + '.' + emb['type']).replace(' ', '') + "/dumps/"
            dump_name = dump_name + create_analogy_dump_name(params, emb, new_format_params)
            if os.path.exists(dump_name):
                with open(dump_name, "rb") as f:
                    test_sol_info = pickle.load(f)
            else:
                # Alternative dump name 
                dump_name = new_format_params["ref_path"] + '_'.join([(model_info[emb['id']]['name'] + '.' + emb['type']).replace(' ', '') for emb in embeddings]) + "/dumps/"
                dump_name = dump_name + create_analogy_dump_name(params, emb, new_format_params)
                if os.path.exists(dump_name):
                    with open(dump_name, "rb") as f:
                        test_sol_info = pickle.load(f)
                else:
                    logger.debug("Dump not found")

            # Encode and format solutions
            formatted_solution = [[model["format"](e) for e in el] for el in test_sol_info["test_question_sol"]]
            solutions = encode_solutions(formatted_solution, tok, strategy=params["multitoken_solutions_strategy"])

            # Add embedding to graph
            ANALOGY_GRAPH_MAP["analogies"](
                model_name, dev_emb, tok, params, solutions, **test_sol_info, color=model["color"]
            )
        
        plt.xlabel('k')
        plt.ylabel('Top-k Accuracy')
        plt.xticks(params["test_k"])

    plt.title(full_title, fontsize=12)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(1))
    plt.grid(linestyle = '--', linewidth = 0.5, which="minor")
    plt.grid(linestyle = '--', linewidth = 1, which="major")
    if new_format_params["save_fig"]:
        plt.savefig(image_name, bbox_inches='tight')
    if new_format_params["show_fig"]:
        plt.show()
    plt.close()