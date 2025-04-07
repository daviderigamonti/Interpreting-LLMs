import logging

from dissecting_llm.src.RQ1_2.load_utils import *
from dissecting_llm.src.RQ1_2.RQ1.constants import *
from dissecting_llm.src.RQ1_2.RQ1.data_utils import *
from dissecting_llm.src.RQ1_2.RQ1.rq1_utils import *


# Logging
setup_logger(LOG_PATH, LOGGER_NAME, "analogies", logging.DEBUG)
logger = logging.getLogger(LOGGER_NAME)


try:

    # Load models
    torch.set_default_device("cpu")

    tokenizers, embeddings = extract_embeddings(
        MODEL_INFO, NEED_KEY,
        rms_output=False, include_in=INCLUDE_INPUT_EMB, include_out=INCLUDE_OUTPUT_EMB, linear_interpolation=LINEAR_INTERPOLATION
    )

    # Models are initially loaded in cpu, then only components that are utilizied are transferred to gpu
    if DEVICE != "cpu":
        # embeddings = [emb | {"emb": emb["emb"].to(DEVICE)}for emb in embeddings]
        torch.set_default_device(DEVICE)

    # Import question-words datasets
    df, _ = import_question_words(
        DATASET_PATHS, DATASET_COLUMN_FORMAT, SELECT_CATEGORIES, FILTER_SINGLE_TOKENS, MODEL_INFO,
        tokenizers=tokenizers
    )

    # Compute sum analogies
    logger.debug("Computing analogies")
    batch_graph_generation(
        MODEL_INFO, tokenizers, embeddings, df, "analogies",
        PARAMETERS, FORMAT_PARAMETERS
    )
    #batch_graph_visualization(
    #    MODEL_INFO, tokenizers, embeddings, df, "analogies",
    #    PARAMETERS, FORMAT_PARAMETERS
    #)
    # Compute delta analogies
    logger.debug("Computing delta analogies")
    batch_graph_generation(
        MODEL_INFO, tokenizers, embeddings, df, "delta_analogies", 
        PARAMETERS_DELTA, FORMAT_PARAMETERS_DELTA
    )

    logger.debug("Completed execution")

except Exception as err:
    logger.error("Execution stopped: %s", repr(err))
finally:
    for handler in logger.handlers:
        handler.close()