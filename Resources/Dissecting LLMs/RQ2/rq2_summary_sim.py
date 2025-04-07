import logging
import pickle
import gc

from numpy import random

from dissecting_llm.src.RQ1_2.load_utils import *
from dissecting_llm.src.RQ1_2.RQ2.constants import *
from dissecting_llm.src.RQ1_2.RQ2.data_utils import *
from dissecting_llm.src.RQ1_2.RQ2.rq2_utils import *


# Logging
setup_logger(LOG_PATH, LOGGER_NAME, "markov", logging.DEBUG)
logger = logging.getLogger(LOGGER_NAME)


try:

    for m, color in zip(
        [
            {"meta-llama/Llama-2-7b-hf": {"name": "LLaMa 2", "format": NOADD}},
            {"mistralai/Mistral-7B-v0.3": {"name": "Mistral v3", "format": NOADD}},
            {"microsoft/Phi-3.5-mini-instruct": {"name": "Phi 3", "format": NOADD}}
        ],
        [
            (0.12, 0.47, 0.71), (0.58, 0.40, 0.74), (0.89, 0.47, 0.76)
        ]
        ):

        # Load models
        torch.set_default_device("cpu")

        model_id = list(m.keys())[0].split("/")[-1] + ( "-rms" if RMS_NORM else "")
        image_path = IMAGE_PATH + "/"
        dump_path = IMAGE_PATH + model_id + "/dumps/"
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)

        logger.debug("Loading: " + model_id)

        tokenizers, embeddings = extract_embeddings(
            m, NEED_KEY, rms_output=RMS_NORM
        )

        if not os.path.exists(image_path):
            os.makedirs(image_path)

        # Models are initially loaded in cpu, then only components that are utilizied are transferred to gpu
        if DEVICE != "cpu":
            embeddings = [emb | {"emb": emb["emb"].to(DEVICE)} for emb in embeddings]
            torch.set_default_device(DEVICE)

        tokenizer = list(tokenizers.values())[0]
        fom_model = {"input": embeddings[0]["emb"], "output": embeddings[1]["emb"]}

        # Build the transition matrix for the FOM, by multiplying input and output embeddings
        fom_trans_matrix = torch.matmul(
            fom_model["input"].weight.data, fom_model["output"].weight.data.T
        )

        fom_trans_matrix = fom_trans_matrix.cpu()
        fom_model["input"] = fom_model["input"].cpu()
        del fom_model["output"]

        markov_trans_matrix = None

        # Self

        # trans_matrix1 = fom_trans_matrix
        # ks_1 = [1, 5, 10, 20, 50, 100]
        # trans_matrix2 = None
        # ks_2 = []
        # ylabel = "Accuracy Against Identity Predictions"
        # title = "Summary_topk_self.pdf"



        # Markov

        del embeddings
        gc.collect()
        torch.cuda.empty_cache()

        n_voc = fom_trans_matrix.size(0)

        # Load wikitext dataset
        raw_wikitext, raw_test_wikitext, df, _ = load_wikitext(tokenizer)
        
        if not os.path.exists(dump_path + "mtm.pt"):
            markov_trans_matrix = build_transition_matrix(df, fom_model["input"].cuda(), smooth=1)
            torch.save(markov_trans_matrix, dump_path + "mtm.pt")
        else:
            markov_trans_matrix = torch.load(dump_path + "mtm.pt")

        markov_trans_matrix = markov_trans_matrix.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        if "df" in locals() or "df" in globals():
            del df
            del raw_wikitext

        trans_matrix1 = fom_trans_matrix
        ks_1 = [1, 5, 10, 20, 50, 100]
        trans_matrix2 = markov_trans_matrix
        ks_2 = [5]
        ylabel = "Accuracy Against Top-$5$ Markov Predictions"
        title = "Summary_topk_markov.pdf"


        # Print

        if trans_matrix2 is not None:
            for k_2 in ks_2:  
                _, reference = next_words(trans_matrix2, k=k_2)
                topk_results = compute_topk_accuracy(trans_matrix1, reference, k=ks_1)
                plt.plot(ks_1, topk_results, color=color, marker='o', label=f'{list(m.values())[0]['name']}Top-$k$ Accuracy')
                plt.xlabel('$k$')
        else:
            reference = list(range(0, trans_matrix1.size(0)))
            topk_results = compute_topk_accuracy(trans_matrix1, reference, k=ks_1)
            plt.plot(ks_1, topk_results, color=color, marker='o', label=f"{list(m.values())[0]['name']} Top-$k$ Accuracy")
            plt.xlabel('$k$')



        del trans_matrix1, trans_matrix2, fom_trans_matrix, markov_trans_matrix

        gc.collect()
        torch.cuda.empty_cache()

    plt.ylabel(ylabel)

    plt.gca().xaxis.set_minor_locator(FixedLocator([15, 30, 40, 66.67, 83.33])) # Print as PDF
    plt.gca().xaxis.set_major_locator(FixedLocator(ks_1)) # Print as PDF

    plt.grid(linestyle = '--', linewidth = 0.5, which="minor")
    plt.grid(linestyle = '--', linewidth = 1, which="major")
    plt.legend(loc="upper left", prop={'size': 9})

    plt.savefig(image_path + title, bbox_inches='tight')
    plt.close()

    logger.debug("Completed execution")

except Exception as err:
    logger.error("Execution stopped: %s", repr(err))
finally:
    for handler in logger.handlers:
        handler.close()