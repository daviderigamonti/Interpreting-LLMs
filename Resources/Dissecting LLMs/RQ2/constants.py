# Formatting functions, used to change the format of inputs to fit models' needs
NOADD = lambda x: x
ADDSPACE = lambda x: " " + x if x[0] != " " else x
ADDALL = lambda x: (x.title(), x.lower())

# Define models
MODEL_INFO = {
    #"meta-llama/Llama-2-7b-hf": {"name": "LLaMa 2", "format": NOADD},
    #"meta-llama/Meta-Llama-3-8B": {"name": "LLaMa 3", "format": NOADD},
    #"tiiuae/falcon-11B": {"name": "Falcon 2", "format": NOADD},
    #"EleutherAI/pythia-6.9b": {"name": "Pythia", "format": NOADD},
    #"mistralai/Mistral-7B-v0.3": {"name": "Mistral v3", "format": NOADD},
    "microsoft/Phi-3.5-mini-instruct": {"name": "Phi 3", "format": NOADD}
}
NEED_KEY = [ "meta-llama/Llama-2-7b-hf", "google/gemma-7b" ]
DEVICE = "cuda"

RMS_NORM = False

IMAGE_PATH = "dissecting_llm/src/RQ1_2/RQ2/img/"

# Logging
LOG_PATH = "dissecting_llm/src/RQ1_2/RQ2/logs/"
LOGGER_NAME = "rq2_logger"