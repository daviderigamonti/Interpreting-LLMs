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
    torch.set_default_device("cpu")
    logger.debug("Distance from FOM to I: %f", distance_from_I(fom_trans_matrix.softmax(dim=-1)))
    torch.set_default_device(DEVICE)

    gc.collect()
    torch.cuda.empty_cache()
    # fom_trans_matrix = fom_trans_matrix.to(DEVICE)

    _, topk = next_words(fom_trans_matrix, k=5)

    words = ["how", "mount", "easy", "hair", "why", "If", "and", "Good"]

    # Encode the test words using the model's tokenizer
    # (only take the first token if the word consists of multiple tokens)
    word_tokens = [tokenizer.encode(word, add_special_tokens=False)[0] for word in words]

    # Visualize likely output words
    for word_token in word_tokens:
        print(vis_topk_result(tokenizer, topk, word_token))

    
    # Transition matrix should predict itself
    _, reference = next_words(fom_trans_matrix, k=1)
    logger.debug("Top-1 accuracy for FOM-FOM prediction: %f", compute_topk_accuracy(fom_trans_matrix, reference, k=1))

    # Top-10 accuracy of the transition matrix modeling the identity matrix
    reference = list(range(0, fom_trans_matrix.size(0)))
    logger.debug("Top-10 accuracy for FOM-I prediction: %f", compute_topk_accuracy(fom_trans_matrix, reference, k=10))

    ks = [1, 5, 10, 20, 50, 100]
    fom_comparison_graph(fom_trans_matrix, ks, image_name=f"{image_path}FOM_topk_self.pdf")

    fom_trans_matrix = fom_trans_matrix.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

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

    torch.set_default_device("cpu")
    logger.debug("Distance from Markov to I: %f", distance_from_I(markov_trans_matrix))
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.set_default_device(DEVICE)

    fom_trans_matrixx = fom_trans_matrix.softmax(dim=-1).to("cuda")
    markov_trans_matrix = markov_trans_matrix.to("cuda")
    fom_trans_matrixx -= markov_trans_matrix 
    d = torch.norm(fom_trans_matrixx,  p="fro")
    logger.debug("Distance from Markov to FOM: %f", d)
    #logger.debug("Distance from Markov to FOM: %f", matrix_distance(fom_trans_matrix.softmax(dim=-1), markov_trans_matrix))

    del d
    del fom_trans_matrixx
    fom_trans_matrix = fom_trans_matrix.to("cpu")
    markov_trans_matrix = markov_trans_matrix.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    # Most probable output 5 words according to FOM model, for each given input across its whole vocabulary
    _, topk = next_words(markov_trans_matrix, k=5)
    # Encode the test words using the model's tokenizer
    # (only take the first token if the word consists of multiple tokens)
    word_tokens = [tokenizer.encode(word, add_special_tokens=False)[0] for word in words]

    # Visualize likely output words
    for word_token in word_tokens:
        print(vis_topk_result(tokenizer, topk, word_token))

    # Markov transition matrix should predict itself
    _, reference = next_words(markov_trans_matrix, k=1)
    logger.debug("Top-1 accuracy for Markov-Markov prediction: %f", compute_topk_accuracy(markov_trans_matrix, reference, k=1))

    # Top-10 accuracy of the Markov transition matrix modeling the identity matrix, this is expected to be very low
    reference = list(range(0, n_voc))
    logger.debug("Top-10 accuracy for Markov-I prediction: %f", compute_topk_accuracy(markov_trans_matrix, reference, k=10))

    # Top-(20, 10) accuracy of the FOM predicting the same as the Markov model
    _, reference = next_words(markov_trans_matrix, k=10)

    logger.debug("Top-20/10 accuracy for FOM-Markov prediction: %f", compute_topk_accuracy(fom_trans_matrix, reference, k=20))

    ks_markov = [1, 10, 20, 30]
    ks_fom = [1, 5, 10, 20, 50, 100]
    fom_comparison_graph(fom_trans_matrix, ks_fom, markov_trans_matrix, ks_markov, image_name=f"{image_path}FOM_topk_Markov.pdf")


    if "df" in locals() or "df" in globals():
        del df
        del raw_wikitext

    fom_trans_matrix = fom_trans_matrix.to(DEVICE)

    gc.collect()
    torch.cuda.empty_cache()


    n_k = [1, 5, 10, 15, 25, 50, 100, 150, 250, 500, 750, 1000, 1500]

    _, fom_set = next_words(fom_trans_matrix, k=max(n_k))
    _, markov_set = next_words(markov_trans_matrix, k=max(n_k))
    random_set = [random.choice(list(range(0, n_voc)), size=max(n_k), replace=False) for j in range(0, n_voc)]

    overlap, overlap_rf, overlap_rm = test_set_sim(fom_set, markov_set, random_set, n_k, overlap_coef)
    jaccard, jaccard_rf, jaccard_rm = test_set_sim(fom_set, markov_set, random_set, n_k, jaccard_sim)
    sorensen, sorensen_rf, sorensen_rm = test_set_sim(fom_set, markov_set, random_set, n_k, sorensen_sim)

    style = {"linestyle": "-", "marker": "", "label":"FOM/Markov", "color": "blue"}
    style_rf = {"linestyle": "--", "marker": "", "label":"FOM/Random", "color": "orange"}
    style_rm = {"linestyle": "--", "marker": "", "label":"Markov/Random", "color": "red"}
    overlap_metrics = [
        {"values": overlap, "style": style},
        {"values": overlap_rf, "style": style_rf},
        {"values": overlap_rm, "style": style_rm}
    ]
    jaccard_metrics = [
        {"values": jaccard, "style": style},
        {"values": jaccard_rf, "style": style_rf},
        {"values": jaccard_rm, "style": style_rm}
    ]
    sorensen_metrics = [
        {"values": sorensen, "style": style},
        {"values": sorensen_rf, "style": style_rf},
        {"values": sorensen_rm, "style": style_rm}
    ]

    graph_test_sim(overlap_metrics, n_k, limit=5, xlabel="k", ylabel="Overlap coefficient", title="Average overlap between vocabulary words predicted by FOM/Markov model", image_name=f"{image_path}overlap5.pdf")
    graph_test_sim(overlap_metrics, n_k, xlabel="k", ylabel="Overlap coefficient", title="Average overlap between vocabulary words predicted by FOM/Markov model", image_name=f"{image_path}overlap.pdf")
    graph_test_sim(jaccard_metrics, n_k, limit=5, xlabel="k",ylabel="Jaccard similarity", title="Average Jaccard similarity between vocabulary words predicted by FOM/Markov model", image_name=f"{image_path}jaccard5.pdf")
    graph_test_sim(jaccard_metrics, n_k, xlabel="k",ylabel="Jaccard similarity", title="Average Jaccard similarity between vocabulary words predicted by FOM/Markov model", image_name=f"{image_path}jaccard.pdf")
    graph_test_sim(sorensen_metrics, n_k, limit=5, xlabel="k",ylabel="Sorensen index", title="Average Sorensen index between vocabulary words predicted by FOM/Markov model", image_name=f"{image_path}sorensen5.pdf")
    graph_test_sim(sorensen_metrics, n_k, xlabel="k",ylabel="Sorensen index", title="Average Sorensen index between vocabulary words predicted by FOM/Markov model", image_name=f"{image_path}sorensen.pdf")

    style = {"linestyle": "", "marker": "",}
    complete_metrics = [
        {"values": overlap, "style": style | {"label":"FOM/Markov Overlap", "color": "blue", "linestyle": "-"}},
        {"values": overlap_rf, "style": style | {"label":"FOM/Random Overlap", "color": "cyan", "linestyle": "--"}},
        {"values": jaccard, "style": style | {"label":"FOM/Markov Jaccard", "color": "red", "linestyle": "-"}},
        {"values": jaccard_rf, "style": style | {"label":"FOM/Random Jaccard", "color": "orange", "linestyle": "--"}},
        {"values": sorensen, "style": style | {"label":"FOM/Markov Sorensen", "color": "green", "linestyle": "-"}},
        {"values": sorensen_rf, "style": style | {"label":"FOM/Random Sorensen", "color": "lightgreen", "linestyle": "--"}},
    ]

    graph_test_sim(complete_metrics, n_k, limit=5, xlabel="k", ylabel="Similarity", title="Average set similarity metrics between vocabulary words predicted by FOM/Markov model", image_name=f"{image_path}overlapmetrics5.pdf")
    graph_test_sim(complete_metrics, n_k, xlabel="k", ylabel="Similarity", title="Average set similarity metrics between vocabulary words predicted by FOM/Markov model", image_name=f"{image_path}overlapmetrics.pdf")

    del overlap, overlap_rf, overlap_rm, jaccard, jaccard_rf, jaccard_rm, sorensen, sorensen_rf, sorensen_rm
    del overlap_metrics, jaccard_metrics, sorensen_metrics, complete_metrics
    del fom_set, markov_set, random_set
    gc.collect()
    torch.cuda.empty_cache()

    torch.set_default_device("cpu")
    I = torch.eye(n_voc)

    perp_dump = dump_path + "perp.pkl"
    if os.path.exists(perp_dump):
        with open(perp_dump, "rb") as f:
            pkl = pickle.load(f)
            perp_markov, perp_fom, perp_I, perp_random = pkl["markov"], pkl["fom"], pkl["I"], pkl["random"] 
    else:
        fom_trans_matrix = fom_trans_matrix.cpu()
        
        perp_markov = compute_perplexity(raw_test_wikitext, tokenizer, lambda tok: markov_trans_matrix[tok, :].log()).cpu()
        markov_trans_matrix = markov_trans_matrix.cpu()
        fom_trans_matrix = fom_trans_matrix.cuda()
        perp_fom = compute_perplexity(raw_test_wikitext, tokenizer, lambda tok: fom_trans_matrix[tok, :]).cpu()
        fom_trans_matrix = fom_trans_matrix.cpu()
        I = I.cuda()
        perp_I = compute_perplexity(raw_test_wikitext, tokenizer, lambda tok: (I[tok, :] + 1e-12).log()).cpu()
        I = I.cpu()
        perp_random = compute_perplexity(raw_test_wikitext, tokenizer, lambda tok: torch.full((n_voc,), 1 / n_voc).log()).cpu()
        
        with open(perp_dump, "w+b") as f:
            pickle.dump({"markov": perp_markov, "fom": perp_fom, "I": perp_I, "random": perp_random}, f)

    style = {"marker": ".", "linestyle": "", "alpha": 0.05, "markersize": 3}
    perp_metrics = [
        {"values": perp_random, "style": style | {"color":"green", "label":"Random Perplexity"}},
        {"values": perp_fom, "style": style | {"color":"blue", "label":"FOM Perplexity"}},
        {"values": perp_markov, "style": style | {"color":"red", "label":"Markov Perplexity"}},
        {"values": perp_I, "style": style | {"color":"pink", "label":"Identity Perplexity"}},
    ]
    graph_test_sim(
        perp_metrics, list(range(0, len(perp_random))),
        xlabel="Sentences", ylabel="Perplexity", plotylim=(25000, 45000),
        title="Perplexity of models over wikitext dataset",
        image_name=f"{image_path}perpmetrics.pdf"
    )

    logger.debug("Mean Markov perplexity: %f", perp_markov.mean())
    logger.debug("Mean FOM perplexity: %f", perp_fom.mean())
    logger.debug("Mean I perplexity: %f", perp_I.mean())
    logger.debug("Mean random perplexity: %f", perp_random.mean())

    owtperp_dump = dump_path + "owtperp.pkl"
    if os.path.exists(owtperp_dump):
        with open(owtperp_dump, "rb") as f:
            pkl = pickle.load(f)
            perp_markov, perp_fom, perp_I, perp_random = pkl["markov"], pkl["fom"], pkl["I"], pkl["random"] 
    else:
        owt = load_dataset("openwebtext", split="train", trust_remote_code=True)
        owt = owt.shuffle(seed=42).select(range(10000))

        I = I.cpu()
        fom_trans_matrix = fom_trans_matrix.cpu()
        markov_trans_matrix = markov_trans_matrix.cuda()
        gc.collect()
        torch.cuda.empty_cache()
        perp_markov = compute_perplexity(owt, tokenizer, lambda tok: markov_trans_matrix[tok, :].log()).cpu()
        markov_trans_matrix = markov_trans_matrix.cpu()
        fom_trans_matrix = fom_trans_matrix.cuda()
        perp_fom = compute_perplexity(owt, tokenizer, lambda tok: fom_trans_matrix[tok, :]).cpu()
        fom_trans_matrix = fom_trans_matrix.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        logI = (I+ 1e-12).log()
        perp_I = compute_perplexity(owt, tokenizer, lambda tok: logI[tok, :]).cpu()
        del logI
        gc.collect()
        torch.cuda.empty_cache()
        logrand = torch.full((n_voc,), 1 / n_voc).log()
        perp_random = compute_perplexity(owt, tokenizer, lambda tok: logrand).cpu()
        del logrand
        gc.collect()
        torch.cuda.empty_cache()

        with open(owtperp_dump, "w+b") as f:
            pickle.dump({"markov": perp_markov, "fom": perp_fom, "I": perp_I, "random": perp_random}, f)

    style = {"marker": ".", "linestyle": "", "alpha": 0.05, "markersize": 3}
    perp_metrics = [
        {"values": perp_random, "style": style | {"color":"green", "label":"Random Perplexity"}},
        {"values": perp_fom, "style": style | {"color":"blue", "label":"FOM Perplexity"}},
        {"values": perp_markov, "style": style | {"color":"red", "label":"Markov Perplexity"}},
        {"values": perp_I, "style": style | {"color":"pink", "label":"Identity Perplexity"}},
    ]
    graph_test_sim(
        perp_metrics, list(range(0, len(perp_random))),
        xlabel="Sentences", ylabel="Perplexity", plotylim=(25000, 45000),
        title="Perplexity of models over wikitext dataset",
        image_name=f"{image_path}perpmetricsowt.pdf"
    )

    logger.debug("Mean Markov perplexity on OWT: %f", perp_markov.mean())
    logger.debug("Mean FOM perplexity on OWT: %f", perp_fom.mean())
    logger.debug("Mean I perplexity on OWT: %f", perp_I.mean())
    logger.debug("Mean random perplexity on OWT: %f", perp_random.mean())


    def kl(a,b):
        return torch.nn.functional.kl_div(
            a.cpu(), b.cpu(), log_target=True, reduction="none", # batchmean
        ).cpu().float()

    I = I.cpu()
    markov_trans_matrix = markov_trans_matrix.cpu()
    fom_trans_matrix = fom_trans_matrix.cpu()

    gc.collect()
    torch.cuda.empty_cache()

    kl_fom_markov = kl(torch.nn.functional.log_softmax(fom_trans_matrix.cuda(), dim=-1), markov_trans_matrix.log())
    kl_markov_fom = kl(markov_trans_matrix.cuda().log(), torch.nn.functional.log_softmax(fom_trans_matrix.cuda(), dim=-1))
    kl_fom_I = kl(torch.nn.functional.log_softmax(fom_trans_matrix.cuda(), dim=-1), (I.cuda() + 1e-12).log())
    kl_I_fom = kl((I.cuda() + 1e-12).log(), torch.nn.functional.log_softmax(fom_trans_matrix.cuda(), dim=-1))
    kl_markov_I = kl(markov_trans_matrix.cuda().log(), (I.cuda() + 1e-12).log())
    gc.collect()
    torch.cuda.empty_cache()
    markov_trans_matrix = markov_trans_matrix.cuda()
    I = I.cuda()
    kl_I_markov = kl((I + 1e-12).log(), markov_trans_matrix.log())

    logger.debug("Mean FOM-Markov KL div: %f", kl_fom_markov.sum(dim=-1).mean(dim=0))
    logger.debug("Mean Markov-FOM KL div: %f", kl_markov_fom.sum(dim=-1).mean(dim=0))
    logger.debug("Mean FOM-I KL div: %f", kl_fom_I.sum(dim=-1).mean(dim=0))
    logger.debug("Mean I-FOM KL div: %f", kl_I_fom.sum(dim=-1).mean(dim=0))
    logger.debug("Mean Markov-I KL div: %f", kl_markov_I.sum(dim=-1).mean(dim=0))
    logger.debug("Mean I-Markov KL div: %f", kl_I_markov.sum(dim=-1).mean(dim=0))

    kl_style = {"marker": ".", "alpha": 0.05, "s": 3}
    kl_metrics = [
        {"values": kl_fom_markov.sum(dim=-1), "style": kl_style | {"color":"blue", "label":"KL FOM-Markov"}},
        {"values": kl_fom_I.sum(dim=-1), "style": kl_style | {"color":"green", "label":"KL FOM-Identity"}},
        {"values": kl_I_markov.sum(dim=-1), "style": kl_style | {"color":"red", "label":"KL Identity-Markov"}},
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
        plt.savefig(f"{image_path}klmetrics.pdf", bbox_inches='tight')
    plt.close()

    logger.debug("Completed execution")

except Exception as err:
    logger.error("Execution stopped: %s", repr(err))
finally:
    for handler in logger.handlers:
        handler.close()