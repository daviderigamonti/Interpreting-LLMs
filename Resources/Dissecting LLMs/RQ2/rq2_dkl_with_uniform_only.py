import logging
import pickle
import gc

from numpy import random

from dissecting_llm.src.RQ1_2.load_utils import *
from dissecting_llm.src.RQ1_2.RQ2.constants import *
from dissecting_llm.src.RQ1_2.RQ2.data_utils import *
from dissecting_llm.src.RQ1_2.RQ2.rq2_utils import *


# Logging
setup_logger(LOG_PATH, LOGGER_NAME, "markov_dkl_uniform", logging.DEBUG)
logger = logging.getLogger(LOGGER_NAME)


try:

    # Load models
    torch.set_default_device("cpu")

    model_id = list(MODEL_INFO.keys())[0].split("/")[-1] + ( "-rms" if RMS_NORM else "")
    image_path = IMAGE_PATH + model_id + "/"
    dump_path = IMAGE_PATH + model_id + "/dumps/"
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    logger.debug("Loading: " + model_id)

    tokenizers, embeddings = extract_embeddings(
        MODEL_INFO, NEED_KEY, rms_output=RMS_NORM
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

    # Delete embeddings to save GPU space
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

    def kl(a,b):
        return torch.nn.functional.kl_div(
            a.cpu(), b.cpu(), log_target=True, reduction="none", # batchmean
        ).cpu().float()

    markov_trans_matrix = markov_trans_matrix.cpu()
    fom_trans_matrix = fom_trans_matrix.cpu()

    gc.collect()
    torch.cuda.empty_cache()

    kl_fom_markov = kl(torch.nn.functional.log_softmax(fom_trans_matrix.cuda(), dim=-1), markov_trans_matrix.log())
    kl_markov_fom = kl(markov_trans_matrix.cuda().log(), torch.nn.functional.log_softmax(fom_trans_matrix.cuda(), dim=-1))
    logger.debug("Mean FOM-Markov KL div: %f", kl_fom_markov.sum(dim=-1).mean(dim=0))
    logger.debug("Mean Markov-FOM KL div: %f", kl_markov_fom.sum(dim=-1).mean(dim=0))

    gc.collect()
    torch.cuda.empty_cache()
    uni = torch.full(fom_trans_matrix.size(), fill_value=1/n_voc).cuda()

    kl_fom_uni = kl(torch.nn.functional.log_softmax(fom_trans_matrix.cuda(), dim=-1), uni.log())
    kl_uni_fom = kl(uni.log(), torch.nn.functional.log_softmax(fom_trans_matrix.cuda(), dim=-1))
    kl_markov_uni = kl(markov_trans_matrix.cuda().log(), uni.log())
    kl_uni_markov = kl(uni.log(), markov_trans_matrix.cuda().log())
    logger.debug("Mean FOM-Uniform KL div: %f", kl_fom_uni.sum(dim=-1).mean(dim=0))
    logger.debug("Mean Uniform-FOM KL div: %f", kl_uni_fom.sum(dim=-1).mean(dim=0))
    logger.debug("Mean Markov-Uniform KL div: %f", kl_markov_uni.sum(dim=-1).mean(dim=0))
    logger.debug("Mean Uniform-Markov KL div: %f", kl_uni_markov.sum(dim=-1).mean(dim=0))

    del uni
    gc.collect()
    torch.cuda.empty_cache()
    I = torch.eye(n_voc).cuda()

    kl_fom_I = kl(torch.nn.functional.log_softmax(fom_trans_matrix.cuda(), dim=-1), (I.cuda() + 1e-12).log())
    kl_I_fom = kl((I.cuda() + 1e-12).log(), torch.nn.functional.log_softmax(fom_trans_matrix.cuda(), dim=-1))
    kl_markov_I = kl(markov_trans_matrix.cuda().log(), (I.cuda() + 1e-12).log())
    kl_I_markov = kl((I + 1e-12).log(), markov_trans_matrix.cuda().log())

    kl_style = {"marker": ".", "alpha": 0.05, "s": 3}
    kl_metrics = [
        {"values": kl_fom_markov.sum(dim=-1), "style": kl_style | {"color":"blue", "label":"KL FOM-Markov"}},
        {"values": kl_fom_I.sum(dim=-1), "style": kl_style | {"color":"green", "label":"KL FOM-Identity"}},
        {"values": kl_fom_uni.sum(dim=-1), "style": kl_style | {"color":"cyan", "label":"KL FOM-Uniform"}},
        {"values": kl_I_markov.sum(dim=-1), "style": kl_style | {"color":"red", "label":"KL Identity-Markov"}},
        {"values": kl_uni_markov.sum(dim=-1), "style": kl_style | {"color":"gold", "label":"KL Uniform-Markov"}},
    ]

    #graph_test_sim(
    #    kl_metrics, list(range(n_voc)),
    #    xlabel="Vocabulary", ylabel="KL-div",
    #    title="KL div over vocabulary",
    #    image_name=f"{image_path}klmetrics.pdf"
    #)

    for metric in kl_metrics:
        plt.scatter(list(range(n_voc)), metric["values"], **metric["style"])

    plt.title("KL div over vocabulary", fontsize=12)

    plt.xlabel("Vocabulary")
    plt.ylabel("KL divergence")

    plt.tick_params(axis="x",which="both", bottom=False, top=False, labelbottom=False)

    legend = plt.legend(prop={'size': 9})
    # Avoid transparency in legend
    for lh in legend.legend_handles: 
        lh.set_alpha(1)
    if f"{image_path}klmetrics.pdf":
        plt.savefig(f"{image_path}klmetricsuniform.pdf", bbox_inches='tight')
    plt.close()

    logger.debug("Completed execution")

except Exception as err:
    logger.error("Execution stopped: %s", repr(err))
finally:
    for handler in logger.handlers:
        handler.close()