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

#args, unknown = parser.parse_known_args()
#numbermodel = args.numbermodel

PARAMETERS = {
    "cos_first_sub_e4": {
        "test_k": [5, 10, 15, 25, 50],
        "k": 50,
        "distance": "cosine",
        "embedding_strategy": "first_only",
        "multitoken_solutions_strategy": "subdivide",
        "pre_normalize": True,
        "layout": {"function": lambda e1, e2, e3, e4: e2 - e1 + e3, "ref": lambda e1, e2, e3, e4: e3, "solution": "e4", "vis": "e2 - e1 + e3 = e4"},
    },
    "cos_avg_sub_e4": {
        "test_k": [5, 10, 15, 25, 50],
        "k": 50,
        "distance": "cosine",
        "embedding_strategy": "average",
        "multitoken_solutions_strategy": "subdivide",
        "pre_normalize": True,
        "layout": {"function": lambda e1, e2, e3, e4: e2 - e1 + e3, "ref": lambda e1, e2, e3, e4: e3, "solution": "e4", "vis": "e2 - e1 + e3 = e4"},
    },
    "cos_first_sub_d": {
        "test_k": [5, 10, 15, 25, 50],
        "k": 50,
        "distance": "cosine",
        "embedding_strategy": "first_only",
        "multitoken_solutions_strategy": "subdivide",
        "pre_normalize": True,
        "layout": {"delta_function": lambda e1, e2, e3, e4, *args: (e4, e3), "pair_function": lambda e4, e3: e3, "sol_function": lambda e4, e3: e4, "vis": "e1 + Δ(e4 - e3) = e2"}
    },
    "cos_avg_sub_d": {
        "test_k": [5, 10, 15, 25, 50],
        "k": 50,
        "distance": "cosine",
        "embedding_strategy": "average",
        "multitoken_solutions_strategy": "subdivide",
        "pre_normalize": True,
        "layout": {"delta_function": lambda e1, e2, e3, e4, *args: (e4, e3), "pair_function": lambda e4, e3: e3, "sol_function": lambda e4, e3: e4, "vis": "e1 + Δ(e4 - e3) = e2"}
    },
}
COLOR_HUE = {
    "-": 0, "input": 0, "output": 0.2,
}
COLOR_HUE = {
    "-": 0, "input": 0, "output": 0.4,
}

numbermodel = ["microsoft/Phi-3.5-mini-instruct"]
inoutmodel = {
    "microsoft/Phi-3.5-mini-instruct": ["input", "output"]
}
parameters = {
    "microsoft/Phi-3.5-mini-instruct": [PARAMETERS["cos_avg_sub_d"], PARAMETERS["cos_avg_sub_e4"]]
}
parameters_colors = {
    "microsoft/Phi-3.5-mini-instruct": [0, 0.2]
}
filter_tokens = False

addtitle = False
addioref = True
addparam = ["layout"]



MODEL_INFO_REF = {
    "word2vec": {"name": "Word2Vec", "format": NOADD, "color": (0.55, 0.34, 0.29)}, #"tab:brown"
    "glove": {"name": "GloVe", "format": NOADD, "color": (0.84, 0.15, 0.17)}, #"tab:red"
    "google-bert/bert-large-uncased": {"name": "BERT", "format": NOADD, "color": (0.17, 0.63, 0.17)}, #"tab:green"
    "gpt2": {"name": "GPT 2", "format": ADDSPACE, "color": (0.74, 0.74, 0.13)}, #"tab:olive"
    "google/gemma-2-2b": {"name": "GEMMA", "format": ADDSPACE, "color": (1.0, 0.49, 0.05)}, #"tab:orange"
    "meta-llama/Llama-2-7b-hf": {"name": "LLaMa 2", "format": NOADD, "color": (0.12, 0.47, 0.71)}, #"tab:blue"
    "meta-llama/Meta-Llama-3-8B": {"name": "LLaMa 3", "format": NOADD, "color": (0.09, 0.75, 0.81)}, #"tab:cyan"
    "mistralai/Mistral-7B-v0.3": {"name": "Mistral v3", "format": NOADD, "color": (0.58, 0.40, 0.74)}, #"tab:purple"
    "microsoft/Phi-3.5-mini-instruct": {"name": "Phi 3", "format": NOADD, "color": (0.89, 0.47, 0.76)} #"tab:pink"
}
MODEL_INFO = {n: MODEL_INFO_REF[n] for n in numbermodel}

MODEL_LABEL_MAP = {
    "Mistral v3": "Mistral v0.3",
    "Phi 3": "Phi 3.5",
}
MODEL_INFO = {k: v | {"labelname": MODEL_LABEL_MAP[v["name"]] if v["name"] in MODEL_LABEL_MAP else v["name"]} for k,v in MODEL_INFO.items()}

# Formatting functions, used to change the format of inputs to fit models' needs
NOADD = lambda x: x
ADDSPACE = lambda x: " " + x if x[0] != " " else x
ADDALL = lambda x: (x.title(), x.lower())

# Define models
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
FILTER_SINGLE_TOKENS = filter_tokens
# BATS import constants
BATS_DIRECTORY = "dissecting_llm/src/data/analogy/bats"
NEW_BATS_FILE = "dissecting_llm/src/data/analogy/bats/bats.txt"
SKIP_CODES = []
CAPITALIZE_CODES = ["E01", "E02", "E03", "E04"]
MAX_SYNONYMS = 3

FORMAT_PARAMETERS = {
    "skip": True,
    "folder_path": "dissecting_llm/src/RQ1_2/RQ1/img/sel_analogies/" + ("single_tokens" if FILTER_SINGLE_TOKENS else "complete") + "/",
    "ref_path": "dissecting_llm/src/RQ1_2/RQ1/img/analogies/" + ("single_tokens" if FILTER_SINGLE_TOKENS else "complete") + "/",
    "show_fig": False,
    "save_fig": True,
    "fig_ext": "pdf",
}
FORMAT_PARAMETERS_DELTA = {
    "skip": True,
    "folder_path": "dissecting_llm/src/RQ1_2/RQ1/img/delta-sel_analogies/" + ("single_tokens" if FILTER_SINGLE_TOKENS else "complete") + "/",
    "ref_path": "dissecting_llm/src/RQ1_2/RQ1/img/delta-analogies/" + ("single_tokens" if FILTER_SINGLE_TOKENS else "complete") + "/",
    "show_fig": False,
    "save_fig": True,
    "fig_ext": "pdf",
}

# Logging
LOG_PATH = "dissecting_llm/src/RQ1_2/RQ1/sel_logs/"
LOGGER_NAME = "rq1_logger"

def colorhue(color, factor):
    return tuple(min(1.0, c * (1 - factor)) for c in color)

# Logging
setup_logger(LOG_PATH, LOGGER_NAME, "delta-sel_analogies", logging.DEBUG)
logger = logging.getLogger(LOGGER_NAME)

# Load models
torch.set_default_device("cpu")

tokenizers, embeddings = extract_embeddings(
    MODEL_INFO, NEED_KEY,
    rms_output=False, include_out=INCLUDE_OUTPUT_EMB, linear_interpolation=LINEAR_INTERPOLATION
)

embeddings = [e for e in embeddings if e["type"] in inoutmodel[e["id"]]]

l = []
for e in embeddings:
    l.extend([e | {
        "parameters": p,
        "color": colorhue(MODEL_INFO[e["id"]]["color"], c),
    } for p, c in zip(parameters[e["id"]], parameters_colors[e["id"]])])
embeddings = l

embeddings = [e | {
    "label": 
        MODEL_INFO[e["id"]]["labelname"] + 
        (" " + e["type"] if addioref else "") + 
        (" " + " ".join(v["vis"] if isinstance(v, dict) else (k if isinstance(v, bool) else str(v)) for k,v in (e["parameters"] | {"type": e["type"]}).items() if k in addparam) if addparam else ""),
    "color": 
        colorhue(e["color"], COLOR_HUE[e["type"]]),
    } for e in embeddings]

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
analogies_type = "sel_analogies"
format_params = FORMAT_PARAMETERS_DELTA


create_log_title_name(model_info, embeddings, {k:[v[0][k] for v in parameters.values()] for k in next(iter(parameters.values()))[0].keys()})
    
new_folder_path = format_params["folder_path"] + create_analogy_models_folder(model_info, embeddings) + "/"
new_format_params = format_params | {
    "folder_path": new_folder_path,
    "ref_path": format_params["ref_path"],
    "norm_ref_path": FORMAT_PARAMETERS["ref_path"],
}

if not os.path.exists(new_format_params["folder_path"]):
    os.makedirs(new_format_params["folder_path"])

params = parameters
params_name = next(iter(params.values()))[0]

image_name = "kacc_" + create_analogy_image_name(params_name, new_format_params)
image_name = new_format_params["folder_path"] + image_name

full_title = create_analogy_title_name(params_name)
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
                torch.nn.Embedding.from_pretrained(torch.nn.functional.normalize(emb["emb"].weight, dim=-1), freeze=True) if emb["parameters"]["pre_normalize"] else emb["emb"]
            ).to(DEVICE) 
        }

        ref_path = new_format_params["ref_path"] if "delta_function" in emb["parameters"]["layout"] else new_format_params["norm_ref_path"]

        # Compute embedding analogy solutions
        dump_name = ref_path + (model_name + '.' + emb['type']).replace(' ', '') + "/dumps/"
        dump_name = dump_name + create_analogy_dump_name(emb["parameters"], emb, new_format_params)
        if os.path.exists(dump_name):
            with open(dump_name, "rb") as f:
                test_sol_info = pickle.load(f)
        else:
            # Alternative dump name 
            dump_name = ref_path + '_'.join([(model_info[emb['id']]['name'] + '.' + emb['type']).replace(' ', '') for emb in embeddings]) + "/dumps/"
            dump_name = dump_name + create_analogy_dump_name(emb["parameters"], emb, new_format_params)
            if os.path.exists(dump_name):
                with open(dump_name, "rb") as f:
                    test_sol_info = pickle.load(f)
            else:
                # Alternative dump name 
                dump_name = ref_path + "_links/" + (model_name + '.' + emb['type']).replace(' ', '') + "/dumps/"
                dump_name = dump_name + create_analogy_dump_name(emb["parameters"], emb, new_format_params)
                if os.path.exists(dump_name):
                    with open(dump_name, "rb") as f:
                        test_sol_info = pickle.load(f)
                else:
                    logger.debug("Dump not found")
                    raise Exception

        # Encode and format solutions
        formatted_solution = [[model["format"](e) for e in el] for el in test_sol_info["test_question_sol"]]
        solutions = encode_solutions(formatted_solution, tok, strategy=emb["parameters"]["multitoken_solutions_strategy"])

        # Add embedding to graph
        ANALOGY_GRAPH_MAP["delta_analogies"](
            model_name, dev_emb, tok, emb["parameters"], solutions, **test_sol_info, 
            color=emb["color"],
            label=dev_emb["label"]
        )

        test_k = emb["parameters"]["test_k"]
    
    plt.xlabel('k')
    plt.ylabel('Top-k Accuracy')
    plt.xticks(test_k)

plt.title(full_title if addtitle else "", fontsize=12)
plt.plot( [], [], alpha=0.5, linestyle='--', c="black", label="Baselines") # Baseline legend
plt.legend(loc="lower right", fancybox=True, framealpha=0.5, prop={'size': 7}, borderpad=0.5, labelspacing=0.6)
plt.gca().xaxis.set_minor_locator(AutoMinorLocator(1))
plt.grid(linestyle = '--', linewidth = 0.5, which="minor")
plt.grid(linestyle = '--', linewidth = 1, which="major")
if new_format_params["save_fig"]:
    plt.savefig(image_name, bbox_inches='tight')
if new_format_params["show_fig"]:
    plt.show()
plt.close()