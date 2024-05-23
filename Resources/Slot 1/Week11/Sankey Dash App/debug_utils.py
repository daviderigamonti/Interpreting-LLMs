# Import packages
import numpy as np
from time import time
import torch
from torch.distributions import Categorical
from enum import Enum

def init_layers(x, emb_type):
    x[0].add_embedding(x[1].squeeze(), emb_type)
    return x[0]

def convert_debug_vectors(debug_vectors, output_len, input_len, emb_type):
    # Input tokens
    temp = [(e[0], e[1].cpu()) for e in debug_vectors[:32]]
    layers = []
    for token in range(input_len):
        for layer_id in range(32):
            layer_n = temp[layer_id][0]
            layer = LayerWrapper(layer_number=layer_n)
            tensor = temp[layer_id][1].select(1, token).squeeze()
            layer.add_embedding(tensor=tensor, value=emb_type)
            layers.append(layer)
    input_layers = np.array(layers).reshape((input_len, 32)).T

    # Output tokens
    temp = [(e[0], e[1].cpu()) for e in debug_vectors[32:]]
    temp_out = [(LayerWrapper(e[0]),e[1])for e in temp]

    temp_out = list(map(init_layers, temp_out, [emb_type]* len(temp_out)))
    out_layers =  np.array(temp_out).reshape((output_len-1, 32)).T
    return np.concatenate([input_layers, out_layers], axis=1)

class EmbeddingTypes(Enum):
    BLOCK_OUTPUT = "block_output"
    BLOCK_INPUT = 'block_input'
    POST_ATTENTION = "post_attention"
    POST_FF = "post_ff"
    POST_ATTENTION_RESIDUAL = "post_attention_residual"

class LayerWrapper:
    def __init__(self, layer_number,):
        self.layer_number = layer_number
        self.probabilities = {}
        self.embeddings = {}
    
    def add_embedding(self, tensor:torch.tensor, value:str):
        self.embeddings[value] = tensor

    def get_embedding(self, emb_type):
        return self.embeddings[emb_type]
    
    def add_probability(self, prob, decoding_strategy:str):
        self.probabilities[decoding_strategy] = prob
    
    def get_probability(self, decoding_strategy):
        return self.probabilities[decoding_strategy]

class Decoder:
    def __init__(self, model, tokenizer):
        self.tokenizer=tokenizer
        self.output_embedding = model.lm_head
        self.input_embedding = model.model.embed_tokens
    
    def _input_embedding_prediction(self, hidden_state):
        """
        """
        output = torch.matmul(hidden_state.squeeze().to(self.input_embedding.weight.device), self.input_embedding.weight.T)
        token_id = output.argmax()
        probabilities = torch.nn.functional.softmax(output)
        return (token_id, probabilities)

    def _output_embedding_prediction(self, hidden_state):
        """
        """
        hidden_state = torch.tensor(hidden_state).to(device)
        logits = self.output_embedding(hidden_state.squeeze())
        logits = logits.float()
        probabilities = torch.nn.functional.softmax(logits)
        pred_id = torch.argmax(logits)
        return (pred_id, probabilities)

    def _interpolated_embedding_prediction(self, hidden_state, layer_n):
        """
        """
        hidden_state = torch.tensor(hidden_state).to(device)
        # Input logits
        input_logits = torch.matmul(hidden_state.squeeze().to(self.input_embedding.weight.device), self.input_embedding.weight.T)

        # Output logits
        output_logits = self.output_embedding(hidden_state.squeeze())
        output_logits = output_logits.float()

        interpolated_embedding = ((33 - layer_n) * (input_logits) + layer_n * (output_logits)) / 33
        
        probabilities = torch.nn.functional.softmax(interpolated_embedding)
        pred_id = torch.argmax(probabilities)
        
        return (pred_id, probabilities)
    
    def decode_hidden_state(self, target_hidden_state, decoding:str, layer: LayerWrapper):
        
        if decoding=='input':
            pred_id, prob =  self._input_embedding_prediction(layer.get_embedding(target_hidden_state))
        elif decoding=='output':
            pred_id, prob =  self._output_embedding_prediction(layer.get_embedding(target_hidden_state))
        elif decoding=='interpolation':
            pred_id, prob =  self._interpolated_embedding_prediction(layer.get_embedding(target_hidden_state), layer.layer_number)
        
        layer.add_probability(round(float(prob[pred_id]) * 100, 2), decoding)
        # Compute entropy 
        entropy = Categorical(probs = prob).entropy()
        layer.add_probability(round(float(entropy) * 100, 2), "entropy")
        return tokenizer.convert_ids_to_tokens([pred_id])[0]
    



##### Layers debug ###########
def preprare_debug(model, generated_output, output_len, input_len):
    """
    Compute all the debugging quantities needed in the experiments

    Returning layers list where each layer contains all the information needed
    """
    input_residual_embedding = model.model.input_residual_embedding
    attention_plus_residual_embedding = model.model.attention_plus_residual_embedding
    post_attention_embedding = model.model.post_attention_embedding
    post_FF_embedding = model.model.post_FF_embedding
    
    # Create a list of LayerWrapper
    layers = []
    emb_type = EmbeddingTypes.BLOCK_OUTPUT
    
    # 1- Prepare matrix of input tokens hidden_state:  N_TOKENS x N_LAYER
    input_hidden_states = generated_output.hidden_states[0]
    
    # Iterate over layers
    for layer_id, layer_tensor in enumerate(input_hidden_states):
        #Iterate over tokens
        per_token_layers = []
        for token_tensor in layer_tensor.squeeze():
            layer = LayerWrapper(layer_number=layer_id)
            layer.add_embedding(token_tensor, emb_type)
            per_token_layers.append(layer)
        layers.append(per_token_layers)
    
    # Aggregate matrix with output
    output_layers = []
    for output_token in generated_output.hidden_states[1:]:
        per_token_layers = []
        for layer_id, tensor in enumerate(output_token):
            layer = LayerWrapper(layer_number=layer_id)
            layer.add_embedding(tensor.squeeze(), emb_type)
            per_token_layers.append(layer)
        output_layers.append(per_token_layers)
    
    # Transpose to have list of: for each layer all tokens
    output_layers = np.array(output_layers).T.tolist()
    
    layers = np.append(layers, output_layers, axis=1)
    emb_type = EmbeddingTypes.POST_ATTENTION
    post_attention = convert_debug_vectors(post_attention_embedding, output_len, input_len, emb_type)
    emb_type = EmbeddingTypes.POST_FF
    post_ff = convert_debug_vectors(post_FF_embedding, output_len, input_len, emb_type)
    emb_type = EmbeddingTypes.POST_ATTENTION_RESIDUAL
    post_att_res = convert_debug_vectors(attention_plus_residual_embedding, output_len, input_len, emb_type)
    emb_type = EmbeddingTypes.BLOCK_INPUT
    in_res = convert_debug_vectors(input_residual_embedding, output_len, input_len, emb_type)


    # Merge other embeddings to the original layers table 
    for layer, post_att in zip(layers[1:], post_attention):
        for l, att in zip(layer, post_att):
            l.add_embedding(att.get_embedding(EmbeddingTypes.POST_ATTENTION), EmbeddingTypes.POST_ATTENTION)
    for layer, p_ff in zip(layers[1:], post_ff):
        for l, ff in zip(layer, p_ff):
            l.add_embedding(ff.get_embedding(EmbeddingTypes.POST_FF), EmbeddingTypes.POST_FF)
    for layer, p_att_res in zip(layers[1:], post_att_res):
        for l, a_r in zip(layer, p_att_res):
            l.add_embedding(a_r.get_embedding(EmbeddingTypes.POST_ATTENTION_RESIDUAL), EmbeddingTypes.POST_ATTENTION_RESIDUAL)
    for layer, i_res in zip(layers[1:], in_res):
        for l, i_r in zip(layer, i_res):
            l.add_embedding(i_r.get_embedding(EmbeddingTypes.BLOCK_INPUT), EmbeddingTypes.BLOCK_INPUT)
    
    # Copy the first layer as the input layer (block_output)
    for layer in layers[0]:
        layer.add_embedding(layer.get_embedding(EmbeddingTypes.BLOCK_OUTPUT), EmbeddingTypes.BLOCK_INPUT)
        layer.add_embedding(layer.get_embedding(EmbeddingTypes.BLOCK_OUTPUT), EmbeddingTypes.POST_ATTENTION)
        layer.add_embedding(layer.get_embedding(EmbeddingTypes.BLOCK_OUTPUT), EmbeddingTypes.POST_ATTENTION_RESIDUAL)
        layer.add_embedding(layer.get_embedding(EmbeddingTypes.BLOCK_OUTPUT), EmbeddingTypes.POST_FF)
    
    # compute residuals contributions percentages

    for layer in layers[0]:    
         layer.add_probability(0.0, "att_res_perc")    
         layer.add_probability(0.0, "ff_res_perc")

    # Compute attention residual percentage
    for row in layers[1:]:
         for single_layer in row:
              initial_residual = single_layer.get_embedding(EmbeddingTypes.BLOCK_INPUT)
              att_emb = single_layer.get_embedding(EmbeddingTypes.POST_ATTENTION)
              contribution = initial_residual.norm(2, dim=-1) / (initial_residual.norm(2, dim=-1) + att_emb.norm(2, dim=-1))
              final_contribition = round(contribution.squeeze().tolist(), 2)
              single_layer.add_probability(final_contribition, "att_res_perc")
    
    # Compute feed forward residual percentage
    for row in layers[1:]:
         for single_layer in row:
              final_residual = single_layer.get_embedding(EmbeddingTypes.POST_FF)
              att_res_emb = single_layer.get_embedding(EmbeddingTypes.POST_ATTENTION_RESIDUAL)
              contribution = att_res_emb.norm(2, dim=-1) / (att_res_emb.norm(2, dim=-1) + final_residual.norm(2, dim=-1))
              final_contribition = round(contribution.squeeze().tolist(), 2)
              single_layer.add_probability(final_contribition, "ff_res_perc")
    return layers
##############################

