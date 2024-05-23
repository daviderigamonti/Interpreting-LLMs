import numpy as np
import pandas as pd
import os
from torch import bfloat16, cuda
from transformers import AutoTokenizer, GenerationConfig, BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM
import accelerate

from models.llama_model import LlamaForCausalLM
from models.mistral_model import MistralForCausalLM

MODEL_MAPPING = {
    "mistralai/Mistral-7B-Instruct-v0.2": MistralForCausalLM,
    "meta-llama/Llama-2-7b-hf": LlamaForCausalLM
}


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_model(model_id:str, quantization:bool=False, device:str=None):
    
    hf_auth=os.environ['HF_TOKEN']

    if device == None:
        device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16,
    )
    model_config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
        output_attentions=True,
        output_hidden_states=True,
        output_scores=True,
        use_auth_token=hf_auth,
    )
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #    model_id,
    #    trust_remote_code=True,
    #    config=model_config,
    #    quantization_config=bnb_config,
    #    device_map='auto',
    #    use_auth_token=hf_auth
    # )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)

    modelClass = MODEL_MAPPING[model_id] if model_id in MODEL_MAPPING else AutoModelForCausalLM 

    if quantization:
            model = modelClass.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map=device,
            token=hf_auth,
            torch_dtype=bfloat16,
        )
    else:
        model = modelClass.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            #quantization_config=bnb_config,
            device_map=device,
            token=hf_auth,
            torch_dtype=bfloat16,
        )

    return model, tokenizer, device, model_config

##### Generate experiment ####

def generate_output(prompt, model, tokenizer, device):

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    input_len = len(inputs.input_ids.squeeze().tolist())
    gen_config = GenerationConfig(
        #output_attentions=True,
        output_hidden_states=True,
        #output_scores=True,
        return_dict_in_generate=True
    )
    generated_output = model.generate(
        inputs.input_ids, generation_config=gen_config
    )
    output_len = len(generated_output.hidden_states)
    return generated_output, output_len, input_len
##############################

##### Decode output ####
def decode_output(output, tokenizer, input_len):
    return tokenizer.decode(output.sequences.squeeze()[input_len:])
##############################

