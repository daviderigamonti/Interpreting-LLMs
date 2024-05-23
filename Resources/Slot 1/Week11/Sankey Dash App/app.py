from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig, StoppingCriteriaList, StoppingCriteria
from collections import defaultdict
from dataclasses import dataclass, field
from experiment_utils import load_model
from itertools import cycle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import dataclasses
import torch
import math

model_id = "meta-llama/Llama-2-7b-hf" # "mistralai/Mistral-7B-v0.1" 
device = "cpu"

torch.set_default_device(device)
model, tokenizer, device, model_config = load_model(model_id=model_id, quantization=False, device=device)

if tokenizer.eos_token_id and not tokenizer.pad_token_id:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

#### https://github.com/oobabooga/text-generation-webui/blob/2cf711f35ec8453d8af818be631cb60447e759e2/modules/callbacks.py#L12
class _SentinelTokenStoppingCriteria(StoppingCriteria):
    def __init__(self, sentinel_token_ids: list, starting_idx: int):
        StoppingCriteria.__init__(self)
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx
        self.shortest = min([x.shape[-1] for x in sentinel_token_ids])

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]
            trimmed_len = trimmed_sample.shape[-1]
            if trimmed_len < self.shortest:
                continue

            for sentinel in self.sentinel_token_ids:
                sentinel_len = sentinel.shape[-1]
                if trimmed_len < sentinel_len:
                    continue

                window = trimmed_sample[-sentinel_len:]
                if torch.all(torch.eq(sentinel, window)):
                    return True

        return False
####

def generate_stopping_criteria(stopgen_tokens, input_len=0):
    return StoppingCriteriaList([
        _SentinelTokenStoppingCriteria(
            sentinel_token_ids = stopgen_tokens,
            starting_idx=input_len
        )
    ])

if model_id in ["microsoft/phi-1_5"]:
    stopgen_tokens = [
        torch.tensor([198, 198]),  # \n\n
        torch.tensor([628])        # \n\n
    ]
    prompt_structure = "Question: {prompt}\n\nAnswer:"
    exclude_token_offset = 3
    fix_characters = [("Ġ", "␣"), ("Ċ", "\n")]
elif model_id in ["meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1"]:
    stopgen_tokens = [
        torch.tensor([1]),  # <s>
        torch.tensor([2])   # </s>
    ]
    prompt_structure = "{prompt}"
    exclude_token_offset = None
    fix_characters = [("<0x0A>", "\n")]

fix_characters += [("\n", "\\n")]


def pad_masked_attentions(attentions, max_len):
    """
    Attention in generative models are masked, we want to plot a heatmap so we must pad all attentions to the same size with 0.0 values
    """
    array_attentions = [np.array(att.float()) for att in attentions] # TODO: optimize
    new_attentions = [np.concatenate([att, np.zeros([max_len - len(att)])]) for att in array_attentions]
    return np.array(new_attentions)

def compute_complete_padded_attentions(generated_output, layer, head):
    single_layer_attentions = []
    # Prompt tokens
    for single_layer_single_head in torch.squeeze(torch.select(generated_output.attentions[0][layer], 1, head)):
        single_layer_attentions.append(single_layer_single_head)
    # Response tokens
    for attentions_per_token in generated_output.attentions[1:]:
        # Take single layer
        single_layer = attentions_per_token[layer]
        # Take only one head
        single_layer_single_head = torch.select(single_layer, 1, head)
        single_layer_attentions.append(single_layer_single_head)
    # Squeeze dimensions to one a one-dimensional tensor
    pure_attentions = [s.squeeze() for s in single_layer_attentions]
    max_seq_len  = len(pure_attentions[-1])
    # Print last attention heatmap
    padded_attentions = pad_masked_attentions(pure_attentions, max_seq_len)
    return padded_attentions

def compute_batch_complete_padded_attentions(generated_output, heads):
    multi_layer_head_attentions = []
    for head in heads:
        multi_layer_attentions = []
        for layer in range(0, len(generated_output.attentions[0])):
            # Prompt tokens
            prompt_att = [
                torch.squeeze(single_head)
                for single_head in torch.squeeze(torch.select(generated_output.attentions[0][layer], 1, head))
            ]
            # Response tokens
            response_att = [
                torch.squeeze(torch.select(single_layer[layer], 1, head))
                for single_layer in generated_output.attentions[1:]
            ]
            # Pad and merge attentions
            multi_layer_attentions.append(pad_masked_attentions( 
                [att_token for att_token in prompt_att + response_att],
                len(response_att[-1])
            ))
        multi_layer_head_attentions.append(multi_layer_attentions)
    return multi_layer_head_attentions

def compute_ids_from_embedding(token_emb, weights, bias, tot_layers=-1, layer_n=-1):
    # Interpolated embeddings
    if type(weights) == dict:
        logits = {k:torch.matmul(token_emb, weight) + bias for k, weight in weights.items()}
        logits = ((tot_layers - layer_n) * (logits["input"]) + layer_n * (logits["output"])) / tot_layers
    # Single embeddings
    else:
        logits = torch.matmul(token_emb, weights) + bias
    return torch.argmax(logits)

def compute_multirep(model, hidden_states, weights, bias, reverse_weights, max_rep=5, tot_layers=-1, layer_n=-1):
    pred_ids = []
    #pred_norms = []
    for n, hs in enumerate(hidden_states):
        tokens = []
        norms = []
        token_emb = hs.squeeze()
        for i in range(0, max_rep):
            # Compute token and embedding norm
            token_id = compute_ids_from_embedding(token_emb, weights, bias, tot_layers=tot_layers, layer_n=n)
            norm = torch.norm(token_emb) 
            # Stop prematurely if norm is too small or if norm is bigger than previous one
            if norm <= 0.01 or (len(norms) > 0 and norm >= norms[-1]):
                break
            # Do not add repreated tokens
            if token_id not in tokens:
                tokens.append(token_id)
            norms.append(norm)
            # Compute next embedding by subtracting the closest embedding to the current embedding
            closest_emb = reverse_weights[token_id]
            token_emb = token_emb - closest_emb
        pred_ids.append(tokens)
        #pred_norms.append(norms)
    return pred_ids#, pred_norms

def test_multirep(model, input, embedding, token=1):
    if embedding == 'output':
        weights = model.lm_head.weight.T
    elif embedding == 'input':
        weights = model.model.embed_tokens.weight.T

    bias = model.lm_head.bias
    if bias:
        reverse_weights = torch.add(weights.T, bias.unsqueeze(dim=1))
    else:
        bias = 0
        reverse_weights = weights.T 
    inputs = tokenizer("Hi, how are you", return_tensors="pt")
    gen_config = GenerationConfig(
        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else None,
        output_attentions=True, output_hidden_states=True, return_dict_in_generate=True
    )
    gen_output = model.generate(inputs.input_ids, generation_config=gen_config, max_new_tokens=5)
    print(tokenizer.decode(gen_output.sequences.squeeze()))
    a,aa = compute_multirep(model, gen_output.hidden_states[1], weights, bias, reverse_weights)
    return [[(tokenizer.decode(c), cc.detach().numpy().tolist()) for c,cc in zip(b,bb)] for b,bb in zip(a,aa)]

def _apply_lm_head(hidden_states, weights, bias, tot_layers=-1):
    """
    Function which takes as input the hidden states of the model and returns the prediction of the next token.
    Uses the language modeling head of input
    """
    pred_ids = []
    for n, token_layer in enumerate(hidden_states):
        token_id = compute_ids_from_embedding(token_layer, weights, bias, tot_layers=tot_layers, layer_n=n)
        pred_ids.append(token_id)
    return pred_ids
    
def embed_hidden_states(model, hidden_states, embedding="output", include_prompt=False, include_end=True, multirep=True, max_rep=10):
    end_idx = len(hidden_states) if include_end else len(hidden_states) - 1
    tot_layers = model.config.num_hidden_layers

    if embedding == 'output':
        weights = model.lm_head.weight.T
        reverse_weights = model.lm_head.weight
    elif embedding == 'input':
        weights = model.model.embed_tokens.weight.T
        reverse_weights = model.model.embed_tokens.weight
    elif embedding == 'interpolate':
        weights = {"input": model.model.embed_tokens.weight.T, "output": model.lm_head.weight.T}
        reverse_weights = {"input": model.model.embed_tokens.weight, "output": model.lm_head.weight}
    else:
        raise ValueError("Embedding not valid")

    bias = 0
    if model.lm_head.bias:
        raise ValueError("Bias not supported") 

    predictions = []
    # Prompt tokens
    if include_prompt:
        for token_states in torch.stack(hidden_states[0]).swapaxes(0, 2):
            if multirep:
                pred_ids = compute_multirep(model, token_states.swapaxes(0, 1), weights, bias, reverse_weights, max_rep=max_rep, tot_layers=tot_layers)
            else:
                pred_ids = [_apply_lm_head(token_states.swapaxes(0, 1), weights, bias, tot_layers=tot_layers)]
            predictions.append([[int(id) for id in idd] for idd in pred_ids])
    # Response tokens
    for token_states in hidden_states[1:end_idx]:
        if multirep:
            pred_ids = compute_multirep(model, token_states, weights, bias, reverse_weights, max_rep=max_rep, tot_layers=tot_layers)
        else:
            pred_ids = [_apply_lm_head(token_states, weights, bias, tot_layers=tot_layers)]
        predictions.append([[int(id) for id in idd] for idd in pred_ids])
    return predictions

def fix_dataframe_characters(df, replacements, multirep=False, columns=False):
    for old, new in replacements:
        df = df.map(lambda x: [i.replace(old, new) for i in x] if multirep else x.replace(old, new))
    if columns:
        for old, new in replacements:
            df.columns = df.columns.str.replace(old, new)
    return df

def extrapolate_debug_info(raw_debug_vector, n_layers):
    new_vector = None
    layer_vector = None
    for iter_tokens in raw_debug_vector:
        i, iter_tokens = iter_tokens
        n_layer = i % n_layers
        layer_vector = torch.cat([layer_vector, iter_tokens], dim=0) if layer_vector != None else iter_tokens
        if n_layer == n_layers - 1:
            new_vector = torch.cat([new_vector, layer_vector], dim=1) if new_vector != None else layer_vector
            layer_vector = None
    new_vector = new_vector.permute([1, 0, 2])
    return new_vector

def extrapolate_debug_vectors(model):
    n_layers = model.config.num_hidden_layers
    debug_vectors = {
        "input_residual_embedding": extrapolate_debug_info(model.model.input_residual_embedding, n_layers),
        "attention_plus_residual_embedding": extrapolate_debug_info(model.model.attention_plus_residual_embedding, n_layers),
        "post_attention_embedding": extrapolate_debug_info(model.model.post_attention_embedding, n_layers),
        "post_FF_embedding": extrapolate_debug_info(model.model.post_FF_embedding, n_layers),
    }
    return debug_vectors

def model_generate(model, tokenizer, prompt, max_extra_length, config, min_stop_length, stopping_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = len(inputs.input_ids.squeeze().tolist())
    max_len = input_len + max_extra_length
    
    gen_config = config
    stopping_criteria = generate_stopping_criteria(stopping_tokens, input_len + min_stop_length)
    
    generated_output = model.generate(inputs.input_ids, generation_config=gen_config, max_length=max_len, stopping_criteria=stopping_criteria)
    outputs = generated_output.sequences.squeeze()
    text_output = tokenizer.decode(generated_output.sequences.squeeze()[input_len:])
    
    all_tokens = tokenizer.convert_ids_to_tokens(generated_output.sequences[0])
    input_tokens = all_tokens[0:input_len]
    generated_tokens = all_tokens[input_len:]
    
    return text_output, generated_output, {"in": input_tokens, "gen": generated_tokens}

def create_hidden_states_df(model, tokenizer, generated_output, gen_tokens, embedding, include_prompt, fix_characters, multirep=False):
    predictions = embed_hidden_states(model, generated_output.hidden_states, embedding, include_prompt=include_prompt, multirep=multirep, max_rep=5)
    rows = [[tokenizer.convert_ids_to_tokens(pred) for pred in pred_list] for pred_list in predictions]
    rows = rows if multirep else np.squeeze(rows)
    if embedding == "input":
        cols = gen_tokens["in"] + gen_tokens["gen"][:-1]
    else:
        cols = gen_tokens["in"][1:] + gen_tokens["gen"]
    df = pd.DataFrame(rows).T.sort_index(ascending=False).rename(columns={n: col for n, col in enumerate(cols)})
    df = fix_dataframe_characters(df, fix_characters, multirep=multirep, columns=True)
    return df


def generate_sankey_from_info(
    df,                                         # Dataframe containing decoded hidden states for nodes
    generated_output,                           # Generated output to retrieve attentions
    res_percent, ffnn_percent,                  # Tensors with shape [n_tokens, n_layers, emb_size] containing input residual percentage
                                                #  and feedforward residual percentage
    model_config,                               # Model config
    sankey_parameters,                          # Sankey diagram parameters
    df_int=None, df_att=None, df_ffnn=None      # Dataframes containing decoded hidden states for intermediate nodes, attention nodes and feedforward nodes
):
    attentions = compute_batch_complete_padded_attentions(generated_output, range(0, model_config.num_attention_heads))[-1]
    linkinfo = {"attentions": attentions, "residuals": res_percent, "ffnn_states": ffnn_percent}
    dfs = {"states": df, "intermediate": df_int, "attention": df_att, "ffnn": df_ffnn}
    sankey_info = generate_sankey(dfs, linkinfo, sankey_parameters)
    fig = format_sankey(*sankey_info, sankey_parameters)
    return fig

@dataclass
class SankeyParameters:
    # DATA
    row_index: int = 0 # Row of starting token (where 0 corresponds to the top row, and n_layers - 1 corresponds to the bottom row)
    token_index: int = 9 # Position index of starting token (where 0 is first token of the input sequence)
    rowlimit: int = 5 # Limit number of layers to visualize
    multirep: bool = False # Accomodate for each token having multiple labels
    show_0: bool = False
    # COLORS
    colormap: list[str, ...] = field( default_factory = lambda : ["#FF6692"] ) # Colors -- colormap = cycle(px.colors.qualitative.Plotly)
    #colormap = cycle(px.colors.qualitative.Plotly)
    color_change_count_threshold: int = 3 # Number of virtual rows that should have the same color associated to them
    color_brightness_range: tuple[float, float] = (-0.5, 0.2) # Brightness range for tokens color gradient
    node_opacity: float = 0.7 # Opacity of nodes
    link_opacity: float = 0.4 # Opacity of links
    non_residual_link_color: tuple[int, int, int] = (100, 100, 100) # Default color for non-resiudal links
    default_node_color: tuple[int, int, int] = (220, 220, 220) # Default color for nodes
    color_nodes: bool = False # If set to true, color nodes based on the colormap, otherwise all nodes will have their default color
    extra_brightness_map: dict[str, float] = field( default_factory = lambda : {"Node": -0.5, "FFNN": 0.15, "Attention": -0.15, "Intermediate": -0.3} )
    # LAYOUT
    print_indexes: bool = False
    rescale_factor: int = 3
    fixed_offsets: dict[str, float] = field( default_factory = lambda : {"Node": 0, "FFNN": 0.02, "Attention": 0.02, "Intermediate": 0} )
    column_pad: float = 0.05
    sankey_zero: float = 0.000000000000001 # Correction to avoid feeding nodes with a coordinate value of 0, which causes problems with Plotly Sankey Diagrams
    size: int = 1200 # Size of square canvas

def cumulative_sankey_traces(
    dfs, linkinfo,             # Dataframes and link info to access labels and node hidden information
    row, indexes, el_indexes, # Dataframe is indexed by index and row, while el_index references the index for sankey visualization
    bases,                    # Base attention value of parents
    labels,                   # Current set of labels for sankey visualization
    elmap,                    # Reference for duplicate nodes as a dictionary indexed with (row, index) and containing a dictionary composed of
                              #  an id and a base
    rowlimit,                 # Depth limit
):
    new_labels = []
    new_indexes = []
    new_elmap = elmap.copy() # TODO: copy necessary?

    under = []
    over = []
    val = []
    types = []
    # Calculate current value of node by weighting its attention value for the parent's weight
    for index, el_index, base in zip(indexes, el_indexes, bases):
        res_w = linkinfo["residuals"][index][-(row + 1)].item()
        res_w += 0.0000000001 if res_w == 0.0 else (-0.0000000001 if res_w == 1.0 else 0) # Prevent 0
        attn_w = 1 - res_w
        mlp_w = linkinfo["ffnn_states"][index][-(row + 1)].item()
        mlp_w += 0.0000000001 if mlp_w == 0.0 else (-0.0000000001 if mlp_w == 1.0 else 0) # Prevent 0
        resattn_w = 1 - mlp_w
        # Create MLP / Attention / Intermediate nodes
        mlp_index = len(new_elmap.keys())
        new_labels.append(dfs["ffnn"].iloc[row+1][index] if dfs["ffnn"] else ["FFNN"])
        new_elmap[(round(row + 1 - 0.8, 2), round(index - 0.5, 2))] = {"id": mlp_index, "base": base * mlp_w, "type": "FFNN"}
        attn_index = len(new_elmap.keys())
        new_labels.append(dfs["attention"].iloc[row+1][index] if dfs["attention"] else ["Attention"])
        new_elmap[(round(row + 1 - 0.45, 2), round(index - 0.5, 2))] = {"id": attn_index, "base": base * attn_w, "type": "Attention"}
        hid_index = len(new_elmap.keys())
        new_labels.append(dfs["intermediate"].iloc[row+1][index] if dfs["intermediate"] else ["-"])
        new_elmap[(round(row + 1 - 0.65, 2), index)] = {"id": hid_index, "base": base, "type": "Intermediate"}
        # Iterate over all elements of the next row
        for i, label in enumerate(dfs["states"].iloc[row+1].tolist()):
            v = base * attn_w * linkinfo["attentions"][row][index][i].item()
            if v > 0:
                over.append(attn_index)
                # If node is already present store its information
                if (row+1, i) in new_elmap:
                    under.append(new_elmap[(row+1, i)]["id"])
                    new_elmap[(row+1, i)]["base"] += v
                # If the node is new create a new entry in the element map with a new sankey index 
                else:
                    new_index = len(new_elmap.keys())
                    new_labels.append(label)
                    new_indexes.append(i)
                    under.append(new_index)
                    new_elmap[(row+1, i)] = {"id": new_index, "base": v, "type": "Node"}
                val.append(v)
                types.append("attention")
        # MLP State
        over.append(el_index)
        under.append(mlp_index)
        val.append(base * mlp_w)
        types.append("mlp")
        over.append(mlp_index)
        under.append(hid_index)
        val.append(base * mlp_w)
        types.append("mlp")
        # Attention State
        over.append(hid_index)
        under.append(attn_index)
        val.append(base * attn_w)
        types.append("att")
        # Residuals
        over.append(hid_index)
        under.append(new_elmap[(row+1, index)]["id"])
        val.append(base * res_w)
        types.append("residual")
        new_elmap[(row+1, index)]["base"] += base * res_w
        over.append(el_index)
        under.append(hid_index)
        val.append(base * resattn_w)
        types.append("residual")
        
    # If depth limit is reached, stop recurring
    if row < rowlimit:
        # Call itself on all the new nodes
        nex_under, nex_over, nex_val, nex_types, nex_labels, new_elmap = cumulative_sankey_traces(
            dfs, linkinfo,
            row+1, new_indexes, [new_elmap[(row+1, i)]["id"] for i in new_indexes],
            [new_elmap[(row+1, i)]["base"] for i in new_indexes],
            new_labels,
            new_elmap,
            rowlimit
        )
        # Update elements map, sankey trace lists and sankey labels list with children's results
        new_labels += nex_labels
        under += nex_under
        over += nex_over
        val += nex_val
        types += nex_types
    # Only executed at topmost level
    if len(el_indexes) == 1 and el_indexes[0] == 0:
        # Complete sankey labels list with starting label
        new_labels = labels + new_labels
    return under, over, val, types, new_labels, new_elmap

# Rescales values of a list inside a given range, if invert is set to True, the range is flipped
def rescale_list(l, range_min=0, range_max=1, old_min=None, old_max=None, invert=False):
    if old_max == None:
        old_max = max(l)
    if old_min == None:
        old_min = min(l)
    old_range = old_max - old_min
    new_range = range_max - range_min

    invert_k = 0
    invert_a = 1
    if invert:
        invert_k = old_max
        invert_a = -1

    return [ range_min + (((invert_k + (invert_a * (el - old_min))) * new_range ) / old_range) for el in l ]

# Given a list and a list of indexes that have been previously sorted, restore the original order of the list
def restore_list_order(l, indexes):
    return [l[indexes.index(i)] for i in range(0, len(indexes))]

# Return a list of RGBA color strings given a list of RGBA colors tuples
def build_rgba_from_tuples(l, opacity=1.0):
    return [f"rgba{tuple(el) + (opacity,)}" if len(el) == 3 else f"rgba{el}" for el in l]

def change_color_brightness(rgb_color, brightness):
    delta_color = tuple([int((channel) * brightness) for channel in rgb_color])
    return tuple([sum(channel) for channel in zip(rgb_color, delta_color)])

def generate_sankey_linkinfo(generated_output, debug_vectors, att_head):
    # TODO: fix range
    attentions = compute_batch_complete_padded_attentions(generated_output, range(0, 32))[att_head] #TODO: fix range
    res_contrib = debug_vectors["input_residual_embedding"]
    attn_contrib = debug_vectors["post_attention_embedding"]
    resattn_contrib = debug_vectors["attention_plus_residual_embedding"]
    ffnn_contrib = debug_vectors["post_FF_embedding"]
    res_percent = []
    resattn_percent = []
    ffnn_percent = []
    for res_token, att_token, resatt_token, ffnn_token in zip(res_contrib, attn_contrib, resattn_contrib, ffnn_contrib):
        res_tokenlayer_list = []
        resatt_tokenlayer_list = []
        ffnn_tokenlayer_list = []
        for res_tokenlayer, att_tokenlayer, resatt_tokenlayer, ffnn_tokenlayer in zip(res_token, att_token, resatt_token, ffnn_token):
            res_tokenlayer_list.append(res_tokenlayer.norm() / (res_tokenlayer.norm() + att_tokenlayer.norm()))
            ffnn_tokenlayer_list.append(ffnn_tokenlayer.norm() / (resatt_tokenlayer.norm() + ffnn_tokenlayer.norm()))
        res_percent.append(res_tokenlayer_list)
        resattn_percent.append(resatt_tokenlayer_list)
        ffnn_percent.append(ffnn_tokenlayer_list)
    linkinfo = {"attentions": attentions, "residuals": res_percent, "ffnn_states": ffnn_percent} # Aggregated intermediate weights information
    return linkinfo

def generate_sankey(df, linkinfo, sankey_parameters: SankeyParameters):
    print("---")
    print(linkinfo["ffnn_states"])
    row_index = sankey_parameters.row_index
    token_index = sankey_parameters.token_index
    token_label = df.iloc[row_index].iloc[token_index]
    dfs = {"states": df, "intermediate": None, "attention": None, "ffnn": None} # TODO
    # Generate diagram data
    under, over, values, types, labels, elmap = cumulative_sankey_traces(
        dfs, linkinfo, 
        row_index, [token_index], [0], 
        [1.0], 
        [token_label], 
        {(row_index, token_index): {"id": 0, "base": 1.0, "base_pow": 1, "type": "Node"}},
        sankey_parameters.rowlimit
    )
    return (under, over, values, types, labels, elmap)

def format_sankey(un, ov, vl, types, lab, elmap, sankey_parameters: SankeyParameters):
    # Handle multiple labels for tokens with multiple representations
    nodes_extra = []
    if sankey_parameters.multirep:
        nodes_extra = [{"text": l} for l in lab]
        lab = [l[0] for l in lab]
    else:
        lab = [np.squeeze(l).item() for l in lab]
        nodes_extra = [{"text": l} for l in lab]

    # Generate numbered labels
    lab = [f"{k[1]} {lab[v['id']]}" if sankey_parameters.print_indexes and el["type"] in ["Node"] else lab[el['id']] for k,el in elmap.items()]

    # Add non-rescaled info to links and nodes extra information
    for k, el in elmap.items():
        nodes_extra[el["id"]] = nodes_extra[el["id"]] | {"v": el["base"]}
    links_extra = [{"v": v, "type": t} for v, t in zip(vl, types)]

    # Rescale node and link values by a rescale factor to fit into graph
    rescale_factor = sankey_parameters.rescale_factor
    rescaled_elmap = {k: el | {"base": el["base"] / rescale_factor } for k,el in elmap.items()}
    rescaled_vl = [el / rescale_factor for el in vl]

    # Create reverse mapping obtaining lists indexed by the node id and containing virtual coordinates and node values
    revmap = [next(k for k,v in rescaled_elmap.items() if v["id"] == i) for i in range(len(rescaled_elmap.keys()))]
    revmap_values = [next(v for k,v in rescaled_elmap.items() if v["id"] == i) for i in range(len(rescaled_elmap.keys()))]
    revmap_x = [key[0] for key in revmap]
    revmap_y = [key[1] for key in revmap]
    # Sort reverse-mapped lists to perform transformations on them with more ease, while keeping an index list to reverse the sorting
    revmap_indexes = [i for i in range(0,len(revmap))]
    revmap_x_sort, revmap_y_sort, revmap_values_sort, revmap_indexes = zip(*sorted(zip(revmap_x, revmap_y, revmap_values, revmap_indexes), key=lambda x: x[0]))

    # Build colors
    node_colors = []
    node_colors_ref = []
    link_colors = []
    colormap = cycle(sankey_parameters.colormap)
    current_color = next(colormap)
    old_x = -1
    change_count = sankey_parameters.color_change_count_threshold
    color_brightness_range = sankey_parameters.color_brightness_range
    # Node colors
    for x, y, v in zip(revmap_x_sort, rescale_list(revmap_y_sort, range_min=color_brightness_range[0], range_max=color_brightness_range[1]), revmap_values_sort):
        # Color switching
        if x != old_x:
            if change_count > sankey_parameters.color_change_count_threshold:
                current_color = next(colormap)
                change_count = 0
            change_count += 1
        color_ref = change_color_brightness(px.colors.hex_to_rgb(current_color), y)
        node_colors_ref.append(color_ref)
        actual_color = sankey_parameters.default_node_color
        if sankey_parameters.color_nodes:
            actual_color = px.colors.hex_to_rgb(current_color)
        color = change_color_brightness(actual_color, y + sankey_parameters.extra_brightness_map[v["type"]])
        node_colors.append(color)
        old_x = x
    node_colors = restore_list_order(node_colors, revmap_indexes)
    node_colors_ref = restore_list_order(node_colors_ref, revmap_indexes)
    # Link colors
    link_colors = [node_colors_ref[el] if typ in ["residual"] else sankey_parameters.non_residual_link_color for typ, el in zip(types, un)]
    # Convert colors and add opacities
    node_colors = build_rgba_from_tuples(node_colors, sankey_parameters.node_opacity)
    link_colors = build_rgba_from_tuples(link_colors, sankey_parameters.link_opacity)

    # Generate columns based on maximum node width for each column to fit nodes into
    col_pad = sankey_parameters.column_pad
    columns_width = [max([v["base"] if y == y_index else 0 for (y, v) in zip(revmap_y_sort, revmap_values_sort)]) for y_index in range(0, 10)] # TODO use actual range
    s = sum(columns_width) + col_pad * len(columns_width)
    columns_width = [w/s + col_pad for w in columns_width]
    columns_ys = []
    tot_w = 0
    for w in columns_width:
        columns_ys.append(tot_w)
        tot_w += w

    # Adjust coordinates 
    revmap_x = rescale_list(revmap_x, range_min=sankey_parameters.sankey_zero, range_max=1, invert=False)
    revmap_y = [ columns_ys[math.ceil(y)] + v["base"] / 2 - sankey_parameters.fixed_offsets[v["type"]] for y, v in zip(revmap_y, revmap_values) ]

    fig = go.Figure(go.Sankey(
        orientation = "v",
        arrangement="fixed",
        valueformat=".5r",
        node=dict(
            customdata=nodes_extra,
            hovertemplate="%{customdata.text}<extra>%{customdata.v:.1%}</extra>",
            align="left",
            label=lab,
            color=node_colors,
            x=revmap_x,
            y=revmap_y,
            pad=800,
        ),
        link=dict(
            customdata=links_extra,
            hovertemplate="%{customdata.type} from %{source.label} to %{target.label} <extra>%{customdata.v:.1%}</extra>",
            source=ov,
            target=un,
            value=rescaled_vl,
            color=link_colors
        )
    ))
    fig.update_layout(
        font_size=12, font_family="Verdana", font_color="black",
        width=sankey_parameters.size, height=sankey_parameters.size,
    )
    return fig


import dash
import diskcache
import uuid

from dash import dcc, html, ctx, Patch, DiskcacheManager
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_daq as daq
import plotly.colors as pc

import plotly.graph_objects as go
import plotly.express as px

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)
app = dash.Dash("Cumulative Sankey Diagram Demo")

colors = {"red": ["#FF6692"], "plotly": px.colors.qualitative.Plotly}

generation_tab = html.Div([
    html.Div([
        dcc.Textarea(id='model_input', placeholder ='Insert prompt...', style={'width': '100%', 'height': "50%"}),
        dcc.Loading(id="model_loading", type="dot", color="#873ba1", children =
                dcc.Textarea(id='model_output', readOnly=True, style={'width': '100%', 'height': "50%"})
            )
    ], style={"float": "left", "width": "70%", "height": "100%", "padding": 2}),
    html.Div([
        html.P(children="# of attention heads to load", style={"margin": "0"}),
        dcc.Dropdown([{"value": i, "label": i+1 if i >= 0 else "Average"} for i in range(-1, model_config.num_attention_heads)], id='model_generate_heads', value=-1, clearable=False),
        dcc.Input(id="min_stop_tokens", type='number', value=1, min=0, max=1024),
        html.Label("Min # tokens for stopping criteria"),
        dcc.Input(id="max_new_tokens", type='number', value=10, min=0, max=1024),
        html.Label("Max # of generated tokens"),
        dcc.Checklist([{"label": "Compute Multiple Token Representations", "value": "multirep"}], value=[], id="multirep_tokens", inline=True),
        html.Button('Generate', id='model_generate', style={"width": "100%", "height": "20px"}),
    ], style={"float": "right", "height": "100%", "width": "20%", "padding": 2}),
], style={"height": "240px"})

vis_data_tab = html.Div([
    html.P("Attention head selector"),
    dcc.Slider(id='attention_heads', marks={}, step=1, value=-1, ),
    html.Div([
        html.P("Embeddings selector"),
        dcc.RadioItems([
            {"label": "Input", "value": "input"},
            {"label": "Output", "value": "output"},
            {"label": "Interpolate", "value": "interpolate"}
        ], id='embeddings', value='interpolate', inline=True),
    ], style={"float": "left", "width": "50%", "height": "100%", "padding": 2}),
    html.Div([
        dcc.Input(id="row_limit", type='number', value=5, min=1, max=31), # TODO: max layers
        html.Label("# layers to visualize"),
        dcc.Input(id="row_index", type='number', value=31, min=0, max=31), #  TODO: max/default layers
        html.Label("Index of starting layer"),
        dcc.Input(id="token_index", type='number', value=0, min=0, max=1024), #  TODO: max index
        html.Label("Index of starting token"),
        dcc.Checklist([{"label": "Show first token", "value": "show"}], id="show_0", value=["show"], inline=True),
    ], style={"float": "right", "width": "20%", "height": "100%", "padding": 2}),
], style={"height": "240px"})

vis_color_tab = html.Div([
    html.Div([
        html.Div([
            html.P("Color mapping"),
            dcc.Dropdown([
                {"value": "red", "label": "Red"},
                {"value": "plotly", "label": "Plotly Palette"}
            ], id='colormap', value="plotly", clearable=False),
            dcc.Input(id="node_opacity", type='number', value=0.7, min=0, max=1, step=0.05), 
            html.Label("Nodes color opacity"),
            dcc.Input(id="link_opacity", type='number', value=0.4, min=0, max=1, step=0.05),
            html.Label("Links color opacity"),
            dcc.Checklist([{"label": "Color nodes", "value": "color"}], id="color_nodes", value=["color"], inline=True),
        ], style={"float": "left", "width": "34%", "height": "100%", "padding": 2}),
        html.Div([
            html.P("Color brightness range"),
            dcc.RangeSlider(-1.0, 1.0, id='color_brightness_range', value=[-0.5, 0.2], allowCross=False, tooltip={"placement": "top", "always_visible": True}),
            dcc.Input(id="extra_bright_node", type='number', value=-0.5, min=-1, max=1, step=0.05), 
            html.Label("Extra brightness for base nodes"), html.Br(),
            dcc.Input(id="extra_bright_ffnn", type='number', value=0.15, min=-1, max=1, step=0.05),
            html.Label("Extra brightness for FFNN nodes"), html.Br(),
            dcc.Input(id="extra_bright_att", type='number', value=-0.15, min=-1, max=1, step=0.05),
            html.Label("Extra brightness for attention nodes"), html.Br(),
            dcc.Input(id="extra_bright_int", type='number', value=-0.3, min=-1, max=1, step=0.05),
            html.Label("Extra brightness for intermediate nodes"),
        ], style={"float": "right", "width": "65%", "height": "100%", "padding": 2}),
    ], style={"float": "left", "width": "49%", "height": "100%", "padding": 2}),
    html.Div([
        html.Div([
            daq.ColorPicker(id="color_mapping_color", label="Color for color-mapping", size=160, value={"hex":"#FF6692"}, labelPosition="bottom"),
        ], style={"float": "right", "width": "30%", "height": "100%", "padding": 2}),
        html.Div([
            daq.ColorPicker(id="non_residual_link_color", label="Color for non-residual links", size=160, value=dict(rgb=dict(r=100, g=100, b=100, a=0)), labelPosition="bottom"),
        ], style={"float": "right", "width": "30%", "height": "100%", "padding": 2}),
        html.Div([
            daq.ColorPicker(id="default_node_color", label="Default color for nodes", size=160, value=dict(rgb=dict(r=220, g=220, b=220, a=0)), labelPosition="bottom"),
        ], style={"float": "right", "width": "30%", "height": "100%", "padding": 2}),
    ], style={"float": "right", "width": "50%", "height": "100%", "padding": 2}),
], style={"height": "240px"})

vis_layout_tab = html.Div([], style={"height": "240px"})

app.layout = html.Div([
    dcc.Tabs(children=[
        dcc.Tab(children=[generation_tab], label="Generation"),
        dcc.Tab(children=[vis_data_tab], label="Visualization (data)"),
        dcc.Tab(children=[vis_color_tab], label="Visualization (colors)"),
        dcc.Tab(children=[vis_layout_tab], label="Visualization (layout)"),
    ],),
    html.Div([
        dcc.Graph(id='sankeyplot'),
    ]),
    dcc.Store(id="run_config"),
    dcc.Store(id="current_run_config"),
    dcc.Store(id="vis_config"),
    dcc.Store(id="notify"),
    dcc.Store(id="graph_id")
])

def rgb_dict_to_tuple(d, a=False):
    tup = (d["r"], d["g"], d["b"])
    tup += (d["a"], ) if a else ()
    return tup

@app.callback(
    Output('run_config', 'data'),
    [
        Input('model_generate_heads', 'value'),
        Input('min_stop_tokens', 'value'),
        Input('max_new_tokens', 'value'),
        Input('multirep_tokens', 'value'),
    ]
)
def update_run_config(gen_heads, min_stop_tokens, max_new_tok, multirep):
    return {
        "gen_heads": gen_heads,
        "min_stop_tokens": min_stop_tokens,
        "max_new_tok": max_new_tok,
        "multirep":len(multirep) > 0,
    }

@app.callback(
    [
        Output('vis_config', 'data'),
        Output("notify", "data"),
    ],
    [
        Input('attention_heads', 'value'), Input('embeddings', 'value'),  Input('row_index', 'value'), Input('token_index', 'value'), Input('row_limit', 'value'),
            Input('show_0', 'value'),
        Input('colormap', 'value'), Input('color_mapping_color', 'value'), Input('color_brightness_range', 'value'), Input('node_opacity', 'value'),
            Input('link_opacity', 'value'), Input('non_residual_link_color', 'value'), Input('default_node_color', 'value'), Input('color_nodes', 'value'),
            Input('extra_bright_node', 'value'), Input('extra_bright_ffnn', 'value'), Input('extra_bright_att', 'value'), Input('extra_bright_int', 'value'),
        
        Input('run_config', 'data')
    ],
    prevent_initial_call=True,
)
def update_vis_config(
    att_head, emb, row_index, token_index, row_limit, show_0,
    colormap, color_mapping_color, color_brightness_range, node_opacity,
        link_opacity, non_residual_link_color, default_node_color, color_nodes,
        extra_bright_node, extra_bright_ffnn, extra_bright_att, extra_bright_int,
    run_config
):
    vis_colormap = colors[colormap] if colormap in colors else [color_mapping_color["hex"]]
    nrlc = rgb_dict_to_tuple(non_residual_link_color["rgb"])
    dnc = rgb_dict_to_tuple(default_node_color["rgb"])
    show_0 = len(show_0) > 0
    color_nodes = len(color_nodes) > 0
    return {
        "head": att_head,
        "embedding": emb,
        "sankey_parameters": dataclasses.asdict(SankeyParameters(
            row_index=row_index, token_index=token_index, rowlimit=row_limit, multirep=run_config["multirep"], show_0=show_0,
            colormap=vis_colormap, color_brightness_range=color_brightness_range, node_opacity=node_opacity,
                link_opacity=link_opacity, non_residual_link_color=nrlc, default_node_color=dnc, color_nodes=color_nodes,
                extra_brightness_map={"Node":extra_bright_node, "FFNN":extra_bright_ffnn, "Attention":extra_bright_att, "Intermediate":extra_bright_int},
        )),
    }, True

@cache.memoize()
def model_output(prompt, session, run_config):
    prompt = prompt_structure.format(prompt=prompt)
    gen_config = GenerationConfig(
        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else None,
        output_attentions=True, output_hidden_states=True, return_dict_in_generate=True
    )
    text_output, generated_output, gen_tokens = model_generate(
            model, tokenizer, prompt, 
            max_extra_length=run_config["max_new_tok"], 
            config=gen_config, 
            min_stop_length=run_config["min_stop_tokens"], stopping_tokens=stopgen_tokens
    )
    debug_vectors = extrapolate_debug_vectors(model)

    dfs = {}
    linkinfos = {}
    for head in range(-1, run_config["gen_heads"] + 1):
        linkinfos[head] = generate_sankey_linkinfo(generated_output, debug_vectors, head)
    for emb in ["input", "output", "interpolate"]:
        dfs[emb] = create_hidden_states_df(
            model, tokenizer, generated_output, gen_tokens, emb, 
            include_prompt=True, fix_characters=fix_characters,
            multirep=run_config["multirep"],
        )
    return text_output, dfs, linkinfos

# Define callback to generate output
@app.callback(
    [
        Output('model_output', 'value'),
        Output('attention_heads', 'marks'),
        Output('attention_heads', 'value'),
        Output('graph_id', 'data'),
        Output('current_run_config', 'data'),
        Output("notify", "data", allow_duplicate=True),
    ],
    Input('model_generate', 'n_clicks'),
    [
        State('model_input', 'value'),
        State('run_config', 'data'),
    ],
    #running=[(Output("model_generate", "disabled"), True, False)],
    prevent_initial_call=True,
    #background=True,
    #manager=background_callback_manager
)
def update_model_generation(click_data, prompt, run_config):

    if ctx.triggered_prop_ids:
        graph_id = str(uuid.uuid4())
        slider_marks = {i: f"Head {i}" for i in range(0, run_config["gen_heads"] + 1)}
        slider_marks.update({-1: "AVG"})
        text_output, _, _ = model_output(prompt, graph_id, run_config)
        return text_output, slider_marks, -1, graph_id, run_config, True
    raise PreventUpdate

@app.callback(
    Output('sankeyplot', 'figure'),
    [
        Input('notify', 'data'),
    ],[
        State('model_input', 'value'),
        State('graph_id', 'data'),
        State('current_run_config', 'data'),
        State('vis_config', 'data'),
    ]
)
def update_sankey_plot(notify, prompt, graph_id, run_config, vis_config):
    if ctx.triggered_prop_ids and run_config and vis_config:
        _, dfs, linkinfos = model_output(prompt, graph_id, run_config)
        sankey_parameters = SankeyParameters(**vis_config["sankey_parameters"])
        cur_linkinfo = linkinfos[vis_config["head"]]
        cur_linkinfo["attentions"] = cur_linkinfo["attentions"] if sankey_parameters.show_0 else [np.array([[0 if i == 0 or j == 0 else e2 for j,e2 in enumerate(e1)] for i,e1 in enumerate(row)]) for row in cur_linkinfo["attentions"]] # Attentions
        sankey_info = generate_sankey(
            dfs[vis_config["embedding"]],
            linkinfos[vis_config["head"]],
            sankey_parameters
        )
        fig = format_sankey(*sankey_info, sankey_parameters)
        return fig
    raise PreventUpdate

if __name__ == "__main__":
    app.run(debug=True, jupyter_mode="_none", port=8050)