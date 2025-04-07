import logging
import time
import sys
import gc
import os

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import normalizers, pre_tokenizers, models, Tokenizer
from gensim.models import KeyedVectors


NON_HF_MODELS = {
    "word2vec": {"path": "dissecting_llm/src/data/GoogleNews-vectors-negative300.bin.gz", "header": True, "cased": True},
    "glove": {"path": "dissecting_llm/src/data/glove.6B.300d.txt", "header": False, "cased": False},
}


def setup_logger(logger_path, logger_name, experiment, level):

    # Make sure logging directory exists
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)

    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    handler_file = logging.FileHandler(logger_path + f"{experiment}-{time.time_ns()}", mode="w")
    handler_out = logging.StreamHandler(sys.stdout)
    handler_file.setFormatter(formatter)
    handler_out.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.addHandler(handler_file)
    logger.addHandler(handler_out)


# Load models and extract their embeddings in a standardized format
def extract_embeddings(model_info, need_key_list, add_normalized=False, rms_output=False, include_in=True, include_out=True, linear_interpolation=list()):
    tokenizers = {}
    embeddings = []
    for model_id, model in model_info.items():
        model_name = model["name"]
        print(f"Loading {model_name}...")
        hf_key=None
        if model_id in need_key_list:
            hf_key = os.environ['HF_TOKEN']
        if model_id in NON_HF_MODELS:
            model = KeyedVectors.load_word2vec_format(
                NON_HF_MODELS[model_id]["path"], 
                binary=NON_HF_MODELS[model_id]["header"], no_header=not NON_HF_MODELS[model_id]["header"]
            )
            input_emb = torch.nn.Embedding.from_pretrained(
                torch.cat([torch.zeros((1, model.vectors.shape[-1])), torch.from_numpy(model.vectors)]),
                freeze=True
            )
            output_emb = input_emb
            tokenizer = Tokenizer(models.WordPiece(
                vocab={key: i for i, key in enumerate(["[UNK]"] + model.index_to_key)},
                unk_token="[UNK]"
            ))
            # Normalization
            norm = [normalizers.NFD()] + ([normalizers.Lowercase()] if not NON_HF_MODELS[model_id]["cased"] else [])
            tokenizer.normalizer = normalizers.Sequence(norm)
            # Pre-tokenization
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            # Fast tokenizer
            tokenizers[model_id] = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        else:
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True, token=hf_key, torch_dtype=torch.bfloat16, attn_implementation="eager",
            )
            tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_key)
            if hf_key:
                del hf_key
            # Extract input and output embeddings
            input_emb = model.get_input_embeddings()
            if "bert" not in model_id: 
                output_emb = get_output_embeddings(model)
            else:
                output_emb = model.cls.predictions.decoder
            if rms_output:
                dtype = input_emb.weight.dtype
                input_emb = torch.nn.Embedding.from_pretrained(
                    (
                        input_emb.weight.to(torch.float32) * torch.rsqrt(input_emb.weight.to(torch.float32).pow(2).mean(-1, keepdim=True) + model.base_model.norm.variance_epsilon)
                    ).to(dtype),
                    freeze=True,
                )
                output_emb = torch.nn.Embedding.from_pretrained(
                    model.base_model.norm.weight * output_emb.weight,
                    freeze=True,
                )
            model_config = model.config
        del model
        torch.cuda.empty_cache()
        gc.collect()
        # Same input and output embeddings
        if torch.all(torch.eq(input_emb.weight.data, output_emb.weight.data)):
            embeddings.append({"id": model_id, "type": "-", "emb": input_emb})
        # Different input and output embeddings
        else:
            if include_in:
                embeddings.append({"id": model_id, "type": "input", "emb": input_emb})
            if include_out:
                embeddings.append({"id": model_id, "type": "output", "emb": output_emb})
            for lin_int in linear_interpolation:
                embeddings.append({
                    "id": model_id, "type": f"interpolated{lin_int}", 
                    "emb": torch.nn.Embedding.from_pretrained(
                        (1 - lin_int / model_config.num_hidden_layers) * input_emb.weight + 
                        (lin_int / model_config.num_hidden_layers) * output_emb.weight
                    )
                })
        print(f"{torch.cuda.memory_allocated(0) / 1024**2} ({torch.cuda.memory_reserved(0) / 1024**2}) / {torch.cuda.get_device_properties(0).total_memory / 1024**2}")
    return tokenizers, embeddings

# Return the output embeddings of a model wrapped in a torch.nn.Embedding object
def get_output_embeddings(model):
    if hasattr(model, "lm_head"):
        weights = model.lm_head.weight
        bias = model.lm_head.bias
    else:
        weights = model.embed_out.weight
        bias = model.embed_out.bias
    # Some models include a bias in their output embeddings, here is discarded
    if bias is None:
        bias = 0
    else:
        print("Warning, bias not utilized")
    return torch.nn.Embedding.from_pretrained(weights, freeze=True)
