# Formatting functions, used to change the format of inputs to fit models' needs
NOADD = lambda x: x
ADDSPACE = lambda x: " " + x if x[0] != " " else x
ADDALL = lambda x: (x.title(), x.lower())

# Define models
MODEL_INFO = {
    #"word2vec": {"name": "Word2Vec", "format": NOADD},
    #"glove": {"name": "GloVe", "format": NOADD},
    #"google-bert/bert-large-uncased": {"name": "BERT", "format": NOADD},
    #"gpt2": {"name": "GPT 2", "format": ADDSPACE},
    #"google/gemma-2-2b": {"name": "GEMMA", "format": ADDSPACE},
    #"meta-llama/Llama-2-7b-hf": {"name": "LLaMa 2", "format": NOADD},
    #"meta-llama/Meta-Llama-3-8B": {"name": "LLaMa 3", "format": NOADD},
    #"mistralai/Mistral-7B-v0.3": {"name": "Mistral v3", "format": NOADD},
    "microsoft/Phi-3.5-mini-instruct": {"name": "Phi 3", "format": NOADD}
}
NEED_KEY = [ "meta-llama/Llama-2-7b-hf", "google/gemma-7b", "meta-llama/Llama-2-13b-hf" ]
DEVICE = "cuda"
INCLUDE_INPUT_EMB = False
INCLUDE_OUTPUT_EMB = False
LINEAR_INTERPOLATION = [7, 15, 23]


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
BATS_CODES_NAMES = {
    "I01": "I01: Noun - Plural Reg",    "D01": "D01: Noun - 'less' Reg", "E01": "E01: Country - Capital",     "L01": "L01: Hypernim - Animal",
    "I02": "I02: Noun - Plural Irreg",  "D02": "D02: 'un' - Adj Reg",    "E02": "E02: Country - Language",    "L02": "L02: Hypernim - Misc",
    "I03": "I03: Adj - Comparative",    "D03": "D03: Adj - 'ly' Reg",    "E03": "E03: UK City - County",      "L03": "L03: Hyponim - Misc",
    "I04": "I04: Adj - Superlative",    "D04": "D04: 'over' - Adj Reg",  "E04": "E04: Name - Nationality",    "L04": "L04: Meronym - Substance",
    "I05": "I05: Verb-Inf - 3Pers",     "D05": "D05: Adj - 'ness' Reg",  "E05": "E05: Name - Occupation",     "L05": "L05: Meronym - Member",
    "I06": "I06: Verb-Inf - Ing",       "D06": "D06: 're' - Verb Reg",   "E06": "E06: Animal - Young",        "L06": "L06: Meronym - Part",
    "I07": "I07: Verb-Inf - Ed",        "D07": "D07: Verb - 'able' Reg", "E07": "E07: Animal - Sound",        "L07": "L07: Synonym - Intensity",
    "I08": "I08: Verb-Ing - 3Pers",     "D08": "D08: Verb - 'er' Reg",   "E08": "E08: Animal - Shelter",      "L08": "L08: Synonym - Exact",
    "I09": "I09: Verb-Ing - Ed",        "D09": "D09: Verb - 'tion' Reg", "E09": "E09: Things - Color",        "L09": "L09: Antonyms - Gradable",
    "I10": "I10: Verb-3Pers - Ed",      "D10": "D10: Verb - 'ment' Reg", "E10": "E10: Male - Female",         "L10": "L10: Antonyms - Binary",
}


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
    "folder_path": "dissecting_llm/src/RQ1_2/RQ1/img/analogies/" + ("single_tokens" if FILTER_SINGLE_TOKENS else "complete") + "/",
    "show_fig": False,
    "save_fig": True,
    "fig_ext": "png",
}
FORMAT_PARAMETERS_DELTA = {
    "skip": True,
    "folder_path": "dissecting_llm/src/RQ1_2/RQ1/img/delta-analogies/" + ("single_tokens" if FILTER_SINGLE_TOKENS else "complete") + "/",
    "show_fig": False,
    "save_fig": True,
    "fig_ext": "png",
}


# Logging
LOG_PATH = "dissecting_llm/src/RQ1_2/RQ1/logs/"
LOGGER_NAME = "rq1_logger"