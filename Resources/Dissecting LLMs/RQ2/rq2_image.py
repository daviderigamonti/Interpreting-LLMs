import logging
import pickle
import gc

from numpy import random

from dissecting_llm.src.RQ1_2.load_utils import *
from dissecting_llm.src.RQ1_2.RQ2.constants import *
from dissecting_llm.src.RQ1_2.RQ2.data_utils import *
from dissecting_llm.src.RQ1_2.RQ2.rq2_utils import *

import numpy as np
import seaborn as sns


# Logging
setup_logger(LOG_PATH, LOGGER_NAME, "markov_image_gen", logging.DEBUG)
logger = logging.getLogger(LOGGER_NAME)

PERP_Y1_LIM = (1e11, 1.05e12)
PERP_Y2_LIM = (29000, 35900)
PERP_Y3_LIM = (0, 2100)
OWTPERP_Y1_LIM = (1e11, 1.05e12)
OWTPERP_Y2_LIM = (28000, 32500)
OWTPERP_Y3_LIM = (0, 10500)

DENSPERP = [
    {"xlim": (0, 3000), "owt_xlim": (0, 7000), "xticks_n": 10},
    None,
    None,
    None,
]



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




    # Top-k Identity

    # ks = [1, 5, 10, 20, 50, 100]
    # fom_comparison_graph(fom_trans_matrix, ks, image_name=f"{image_path}FOM_topk_self.pdf")




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




    # Top-k Markov

    # ks_markov = [1, 10, 20, 30]
    # ks_fom = [1, 5, 10, 20, 50, 100]
    # fom_comparison_graph(fom_trans_matrix, ks_fom, markov_trans_matrix, ks_markov, image_name=f"{image_path}FOM_topk_Markov.pdf")




    # Overlap Metrics

    # n_k = [1, 5, 10, 15, 25, 50, 100, 150, 250, 500, 750, 1000, 1500]

    # _, fom_set = next_words(fom_trans_matrix, k=max(n_k))
    # _, markov_set = next_words(markov_trans_matrix, k=max(n_k))
    # random_set = [random.choice(list(range(0, n_voc)), size=max(n_k), replace=False) for j in range(0, n_voc)]

    # overlap, overlap_rf, overlap_rm = test_set_sim(fom_set, markov_set, random_set, n_k, overlap_coef)
    # jaccard, jaccard_rf, jaccard_rm = test_set_sim(fom_set, markov_set, random_set, n_k, jaccard_sim)

    # style = {"linestyle": "-", "marker": "", "label":"FOM/Markov", "color": "blue"}
    # style_rf = {"linestyle": "--", "marker": "", "label":"FOM/Random", "color": "orange"}
    # style_rm = {"linestyle": "--", "marker": "", "label":"Markov/Random", "color": "red"}
    # overlap_metrics = [
    #     {"values": overlap, "style": style},
    #     {"values": overlap_rf, "style": style_rf},
    #     {"values": overlap_rm, "style": style_rm}
    # ]
    # jaccard_metrics = [
    #     {"values": jaccard, "style": style},
    #     {"values": jaccard_rf, "style": style_rf},
    #     {"values": jaccard_rm, "style": style_rm}
    # ]

    # graph_test_sim(overlap_metrics, n_k, limit=5, xlabel="$k$", ylabel="Overlap coefficient", image_name=f"{image_path}overlap5.pdf")
    # graph_test_sim(overlap_metrics, n_k, xlabel="$k$", ylabel="Overlap coefficient", image_name=f"{image_path}overlap.pdf")
    # graph_test_sim(jaccard_metrics, n_k, limit=5, xlabel="$k$",ylabel="Jaccard similarity", image_name=f"{image_path}jaccard5.pdf")
    # graph_test_sim(jaccard_metrics, n_k, xlabel="$k$",ylabel="Jaccard similarity", image_name=f"{image_path}jaccard.pdf")

    # style = {"linestyle": "", "marker": "",}
    # complete_metrics = [
    #     {"values": overlap, "style": style | {"label":"FOM/Markov Overlap", "color": "blue", "linestyle": "-"}},
    #     {"values": overlap_rm, "style": style | {"label":"Random/Markov Overlap", "color": "green", "linestyle": "--", "alpha": 0.5}},
    #     {"values": overlap_rf, "style": style | {"label":"FOM/Random Overlap", "color": "cyan", "linestyle": "--", "alpha": 0.5}},
    #     {"values": jaccard, "style": style | {"label":"FOM/Markov Jaccard", "color": "red", "linestyle": "-"}},
    #     {"values": jaccard_rm, "style": style | {"label":"Random/Markov Jaccard", "color": "yellow", "linestyle": "--", "alpha": 0.5}},
    #     {"values": jaccard_rf, "style": style | {"label":"FOM/Random Jaccard", "color": "orange", "linestyle": "--", "alpha": 0.5}},
    # ]

    # graph_test_sim(complete_metrics, n_k, limit=5, xlabel="$k$", ylabel="Similarity", image_name=f"{image_path}overlapmetrics5.pdf")
    # graph_test_sim(complete_metrics, n_k, xlabel="$k$", ylabel="Similarity", image_name=f"{image_path}overlapmetrics.pdf")




    # Perplexity

    I = torch.eye(n_voc)

    perp_dump = dump_path + "perp.pkl"
    with open(perp_dump, "rb") as f:
        pkl = pickle.load(f)
        perp_markov, perp_fom, perp_I, perp_random = pkl["markov"], pkl["fom"], pkl["I"], pkl["random"] 

    style = {"marker": ".", "linestyle": "", "alpha": 0.1, "markersize": 3}
    perp_metrics = [
        {"values": perp_markov, "style": style | {"color":"red", "label":"Markov Perplexity"}},
        {"values": perp_fom, "style": style | {"color":"#1252c9", "label":"FOM Perplexity"}},
        {"values": perp_random, "style": style | {"color":"#28c936", "label":"Random Perplexity"}},
        {"values": perp_I, "style": style | {"color":"#6e5b1d", "label":"Identity Perplexity"}},
    ]
    
    n_k = list(range(0, len(perp_random)))
    n_metrics = len(perp_metrics)
    order = [list(range(n_metrics)) for i in n_k]

    # f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, facecolor='w', figsize=(7.2, 6.2))

    # plt.subplots_adjust(hspace=0.06)

    # for i in n_k:
    #     random.shuffle(order[i])
    # for i in range(n_metrics):
    #     order_i = [[n for n,l in enumerate(order) if l[j] == i] for j in range(n_metrics)]
    #     for n, metric in enumerate(perp_metrics):
    #         ax1.plot(order_i[n], [metric["values"][j] for j in order_i[n]], **metric["style"] | {"label": ""})
    #         ax2.plot(order_i[n], [metric["values"][j] for j in order_i[n]], **metric["style"] | {"label": ""})
    #         ax3.plot(order_i[n], [metric["values"][j] for j in order_i[n]], **metric["style"] | {"label": ""})

    # f.supxlabel("WikiText Sentences")
    # f.supylabel("Perplexity")

    # ax1.grid(linestyle = '--', linewidth = 0.5, which="minor")
    # ax2.grid(linestyle = '--', linewidth = 0.5, which="minor")
    # ax3.grid(linestyle = '--', linewidth = 0.5, which="minor")
    # ax1.grid(linestyle = '--', linewidth = 1, which="major")
    # ax2.grid(linestyle = '--', linewidth = 1, which="major")
    # ax3.grid(linestyle = '--', linewidth = 1, which="major")
    # ax1.grid(visible=False, axis="x")
    # ax2.grid(visible=False, axis="x")
    # ax3.grid(visible=False, axis="x")

    # ax1.spines['bottom'].set_visible(False)
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)
    # ax3.spines['top'].set_visible(False)

    # ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1e}'))
    # ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    # ax2.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    # ax3.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    # d = .35  # proportion of vertical to horizontal extent of the slanted line
    # slant = dict(marker=[(-1, -d), (1, d)], markersize=12,linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    # ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **slant)
    # ax2.plot([0, 1, 0, 1], [0, 0, 1, 1], transform=ax2.transAxes, **slant)
    # ax3.plot([0, 1], [1, 1], transform=ax3.transAxes, **slant)

    # ax1.set_ylim(PERP_Y1_LIM)
    # ax2.set_ylim(PERP_Y2_LIM)
    # ax3.set_ylim(PERP_Y3_LIM)

    # for metric in perp_metrics:
    #     plt.plot(-1, -1, **metric["style"])

    # legend = plt.legend(prop={'size': 9}, loc="right")
    # # Avoid transparency in legend
    # for lh in legend.legend_handles: 
    #     lh.set_alpha(1)

    # plt.savefig(f"{image_path}perpmetrics.pdf", bbox_inches='tight')

    # plt.close(f)

    f, axes = plt.subplots(1, 4, facecolor='w', figsize=(26, 6))
    for metric, ax, densperp in zip(perp_metrics, axes, DENSPERP):
        sns.histplot(pd.DataFrame({metric["style"]["label"]: metric["values"]}), ax=ax, palette=[metric["style"]["color"]], stat="proportion", kde=True, alpha=0.3, bins=150 if ax == axes[0] else "auto")
        plt.setp(ax.patches, linewidth=0.4, edgecolor=(metric["style"]["color"], 0.35))
        ax.set_xlabel("Perplexity")
        ax.set_ylabel("Density Proportion")
        ax.grid(linestyle = '--', linewidth = 0.5, which="minor")
        ax.grid(linestyle = '--', linewidth = 1, which="major")
        ticks = ax.get_xticks()
        if max(ticks) - min(ticks) < 1:
            offset = (max(ticks) - min(ticks)) / 20
            ax.set_xlim([min(ticks) - offset, max(ticks) + offset])
        ax.ticklabel_format(useOffset=False)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=30, ha="right")
        if densperp:
            ticks = [
                densperp["xlim"][0] + int(i*((densperp["xlim"][1]-densperp["xlim"][0])/densperp["xticks_n"]))
                for i in range(densperp["xticks_n"]+1)
            ]
            ax.set_xticks(ticks, labels=ticks)
            ax.set_xlim([densperp["xlim"][0], densperp["xlim"][1]])
    iticks = axes[-1].get_xticks()
    iticks[0] = min(perp_metrics[-1]["values"])
    axes[-1].set_xticks(iticks, labels=iticks)
    axes[-1].set_xlim([min(perp_metrics[-1]["values"]), None])
    axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}" if x > 1e10 else f"{x:.0f}"))

    plt.savefig(f"{image_path}perpdensity.pdf", bbox_inches='tight')
    plt.close(f)



    owtperp_dump = dump_path + "owtperp.pkl"
    with open(owtperp_dump, "rb") as f:
        pkl = pickle.load(f)
        perp_markov, perp_fom, perp_I, perp_random = pkl["markov"], pkl["fom"], pkl["I"], pkl["random"]

    style = {"marker": ".", "linestyle": "", "alpha": 0.1, "markersize": 3}
    perp_metrics = [
        {"values": perp_markov, "style": style | {"color":"red", "label":"Markov Perplexity"}},
        {"values": perp_fom, "style": style | {"color":"#1252c9", "label":"FOM Perplexity"}},
        {"values": perp_random, "style": style | {"color":"#28c936", "label":"Random Perplexity"}},
        {"values": perp_I, "style": style | {"color":"#6e5b1d", "label":"Identity Perplexity"}},
    ]
    n_k = list(range(0, len(perp_random)))
    n_metrics = len(perp_metrics)
    order = [list(range(n_metrics)) for i in n_k]

    # f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, facecolor='w', figsize=(7.2, 6.2))

    # plt.subplots_adjust(hspace=0.06)

    # for i in n_k:
    #     random.shuffle(order[i])
    # for i in range(n_metrics):
    #     order_i = [[n for n,l in enumerate(order) if l[j] == i] for j in range(n_metrics)]
    #     for n, metric in enumerate(perp_metrics):
    #         ax1.plot(order_i[n], [metric["values"][j] for j in order_i[n]], **metric["style"] | {"label": ""})
    #         ax2.plot(order_i[n], [metric["values"][j] for j in order_i[n]], **metric["style"] | {"label": ""})
    #         ax3.plot(order_i[n], [metric["values"][j] for j in order_i[n]], **metric["style"] | {"label": ""})

    # f.supxlabel("WikiText Sentences")
    # f.supylabel("Perplexity")

    # ax1.grid(linestyle = '--', linewidth = 0.5, which="minor")
    # ax2.grid(linestyle = '--', linewidth = 0.5, which="minor")
    # ax3.grid(linestyle = '--', linewidth = 0.5, which="minor")
    # ax1.grid(linestyle = '--', linewidth = 1, which="major")
    # ax2.grid(linestyle = '--', linewidth = 1, which="major")
    # ax3.grid(linestyle = '--', linewidth = 1, which="major")
    # ax1.grid(visible=False, axis="x")
    # ax2.grid(visible=False, axis="x")
    # ax3.grid(visible=False, axis="x")

    # ax1.spines['bottom'].set_visible(False)
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)
    # ax3.spines['top'].set_visible(False)

    # ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1e}'))
    # ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    # ax2.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    # ax3.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    # d = .35  # proportion of vertical to horizontal extent of the slanted line
    # slant = dict(marker=[(-1, -d), (1, d)], markersize=12,linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    # ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **slant)
    # ax2.plot([0, 1, 0, 1], [0, 0, 1, 1], transform=ax2.transAxes, **slant)
    # ax3.plot([0, 1], [1, 1], transform=ax3.transAxes, **slant)

    # ax1.set_ylim(OWTPERP_Y1_LIM)
    # ax2.set_ylim(OWTPERP_Y2_LIM)
    # ax3.set_ylim(OWTPERP_Y3_LIM)

    # for metric in perp_metrics:
    #     plt.plot(-1, -1, **metric["style"])

    # legend = plt.legend(prop={'size': 9}, loc="right")
    # # Avoid transparency in legend
    # for lh in legend.legend_handles: 
    #     lh.set_alpha(1)

    # plt.savefig(f"{image_path}owtperpmetrics.pdf", bbox_inches='tight')

    # plt.close(f)

    f, axes = plt.subplots(1, 4, facecolor='w', figsize=(26, 6))
    for metric, ax, densperp in zip(perp_metrics, axes, DENSPERP):
        sns.histplot(pd.DataFrame({metric["style"]["label"]: metric["values"]}), ax=ax, palette=[metric["style"]["color"]], stat="proportion", kde=True, alpha=0.3, bins=150 if ax == axes[0] else "auto")
        plt.setp(ax.patches, linewidth=0.4, edgecolor=(metric["style"]["color"], 0.35))
        ax.set_xlabel("Perplexity")
        ax.set_ylabel("Density Proportion")
        ax.grid(linestyle = '--', linewidth = 0.5, which="minor")
        ax.grid(linestyle = '--', linewidth = 1, which="major")
        ticks = ax.get_xticks()
        if max(ticks) - min(ticks) < 1:
            offset = (max(ticks) - min(ticks)) / 20
            ax.set_xlim([min(ticks) - offset, max(ticks) + offset])
        ax.ticklabel_format(useOffset=False)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=30, ha="right")
        if densperp:
            ticks = [
                densperp["owt_xlim"][0] + int(i*((densperp["owt_xlim"][1]-densperp["owt_xlim"][0])/densperp["xticks_n"]))
                for i in range(densperp["xticks_n"]+1)
            ]
            ax.set_xticks(ticks, labels=ticks)
            ax.set_xlim([densperp["owt_xlim"][0], densperp["owt_xlim"][1]])
    iticks = axes[-1].get_xticks()
    iticks[0] = min(perp_metrics[-1]["values"])
    axes[-1].set_xticks(iticks, labels=iticks)
    axes[-1].set_xlim([min(perp_metrics[-1]["values"]), None])
    axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}" if x > 1e10 else f"{x:.0f}"))

    plt.savefig(f"{image_path}owtperpdensity.pdf", bbox_inches='tight')
    plt.close(f)

    logger.debug("Completed execution")

except Exception as err:
    logger.error("Execution stopped: %s", repr(err))
finally:
    for handler in logger.handlers:
        handler.close()