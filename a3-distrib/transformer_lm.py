# models.py

import numpy as np
from transformer import *
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)
    
#=======================================================NN.encoder Model==================================================================
# NN NLM
# class NeuralLanguageModel(LanguageModel):
#     def __init__(self, model, vocab_index):
#         self.model = model
#         self.vocab_index = vocab_index
        
#     def get_next_char_log_probs(self, context):
#         self.model.eval()  

#         indices = [self.vocab_index.index_of(char) for char in context]

#         with torch.no_grad():
#             input_tensor = torch.tensor(indices, dtype=torch.long).view(1, -1)
#             output = self.model(input_tensor)
            
#             if output.size(1) == 0:
#                 # If the model did not produce any output, return a uniform distribution
#                 return np.ones([len(self.vocab_index)]) * np.log(1.0 / len(self.vocab_index))
            
#             output_probs = F.log_softmax(output[0, -1, :], dim=-1)
#         # print("output_probs:", output_probs)
#         return output_probs.numpy()

#     def get_log_prob_sequence(self, next_chars, context):
#         log_probs = 0.0
#         # print(context)
        
        
#         # for i in range(len(next_chars)):
#         #     next_char_log_probs = self.get_next_char_log_probs(context)
#         #     log_probs += next_char_log_probs[self.vocab_index.index_of(next_chars[i])]
#         #     context += next_chars[i]
#         #     context = context[i+1:] #right shift 1 index
            
#         for char in next_chars:
#             next_char_log_probs = self.get_next_char_log_probs(context)
#             # print(next_char_log_probs)
#             log_probs += next_char_log_probs[self.vocab_index.index_of(char)]
#             context += char
#         # print("log_probs:", log_probs)
#         return log_probs
    

# # nn.Encoder Model
# class LM(nn.Module):
#     def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, num_positions):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.positional_encoding = PositionalEncoding(d_model, num_positions)

#         encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         self.fc = nn.Linear(d_model, vocab_size)

#     def forward(self, sequence):

#         embeddings = self.embedding(sequence)
#         positional_encoded = self.positional_encoding(embeddings)
        
#         # Mask
#         seq_length = sequence.size(0)
#         mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
#         mask = mask.masked_fill(mask == 1, float('-inf'))
#         # mask = nn.Transformer.generate_square_subsequent_mask(seq_length)
        
#         transformer_output = self.transformer_encoder(positional_encoded, mask=mask)
        
#         output = self.fc(transformer_output)

#         output = F.softmax(output, dim=-1)
#         # print(output)
#         return output
    
# # nn.Encoder train function
# def train_lm(args, train_text, dev_text, vocab_index):
#     """
#     :param args: command-line args, passed through here for your convenience
#     :param train_text: train text as a sequence of characters
#     :param dev_text: dev text as a sequence of characters
#     :param vocab_index: an Indexer of the character vocabulary (27 characters)
#     :return: a NeuralLanguageModel instance trained on the given data
#     """

#     #Initialize the language model
#     model = LM(vocab_size=len(vocab_index), d_model=200, nhead=1, num_layers=6, dim_feedforward=200, num_positions=512)
    
    
#     # Define loss function and optimizer
#     loss_fc = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.00001)

#     batch_size = 32
#     # seq_len = 20  # chunk??
#     num_batches = len(train_text) // batch_size 

#     num_epochs = 2

#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0.0

#         for batch_idx in range(num_batches-1):
#             start_idx = batch_idx * batch_size 
#             end_idx = (batch_idx + 1) * batch_size 

#             # batch_data
#             batch_data = train_text[start_idx:end_idx]
#             indices = [vocab_index.index_of(char) for char in batch_data]
#             batch_tensor = torch.tensor(indices, dtype=torch.long)
#             batch_tensor = batch_tensor.view(batch_size, -1)

#             # target 
#             target_char = train_text[start_idx + 1 : end_idx + 1] 
#             target_indices = torch.tensor([vocab_index.index_of(char) for char in target_char], dtype=torch.long)
            
#             # model
#             output = model(batch_tensor)
#             # print(output.size())
#             output_flat = output.view(-1, 27)
#             # output_flat = output.view(-1, 27)
#             # print(log_probs.size()) # 32, 20, 27
#             # print(output_flat.size(), target_indices.size())
            
#             loss = loss_fc(output_flat, target_indices)

#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         avg_loss = total_loss / num_batches
#         print(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}")
        
#         neural_lm = NeuralLanguageModel(model, vocab_index)

#     return neural_lm

# ====================================================Part 1 Model=======================================================================
# Part 1 Model
class NeuralLanguageModel(LanguageModel):
    def __init__(self, model, vocab_index):
        self.model = model
        self.vocab_index = vocab_index
        self.model.train(False)

    def get_next_char_log_probs(self, context):
        idxs = [self.vocab_index.index_of(char) for char in context]
        idxs = torch.tensor(idxs, dtype=torch.long)
        probs = self.model(idxs.unsqueeze(0))
        probs = probs.squeeze(0)
        probs = probs[len(context)-1]
        
        return probs.detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        log_prob = 0.0
        m = len(next_chars)
        context = list(context + next_chars[0:m-1])
        n = len(context)
        
        idxs = [self.vocab_index.index_of(char) for char in context]
        idxs = torch.tensor(idxs, dtype=torch.long)
        
        probs = self.model(idxs.unsqueeze(0))
        probs = probs.squeeze(0)

        i = n - m
        for char in next_chars:
            log_prob += probs[i][self.vocab_index.index_of(char)]
            i += 1
        # print(log_prob)
        
        return log_prob.item()
    
#Part 1 Transformer
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions)
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, num_classes)
        
        # raise Exception("Implement me")

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        embeddings = self.embedding(indices)
        positional_encoded = self.positional_encoding(embeddings)
        
        
        for layer in self.layers:
            positional_encoded = layer(positional_encoded)
            
        
        logits = self.fc(positional_encoded)
        
        log_probs = F.log_softmax(logits, dim=-1)
        # log_probs = F.softmax(logits, dim=-1)
        
        return log_probs

# Part 1 TransformerLayer
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        
        # Self-Attention components
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        # self.output_linear = nn.Linear(d_internal, d_model)
        
        # Add Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward Layer
        self.linear1 = nn.Linear(d_model, d_internal)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_internal, d_model)
        
        # raise Exception("Implement me")
    
    def scaled_dot_product_attention(self, q, k, v):
        dk = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))
        
        # Create the upper triangular mask with -inf
        upper_triangular_mask = torch.triu(torch.full((scores.size(-2), scores.size(-1)), float('-inf')), diagonal=1)
        # Create the lower triangular mask with 1
        lower_triangular_mask = torch.tril(torch.zeros((scores.size(-2), scores.size(-1))), diagonal=-1)  
        
        mask = upper_triangular_mask + lower_triangular_mask
        scores = scores + mask
        
        attention_weights = F.softmax(scores, dim=-1)
        
        z = torch.matmul(attention_weights, v)
        
        self.attention_map = attention_weights
        
        return z

    def forward(self, input_vecs):
        q = self.query_linear(input_vecs)
        k = self.key_linear(input_vecs)
        v = self.key_linear(input_vecs)
        
        attention_output = self.scaled_dot_product_attention(q, k, v)
        
        # attention_output = self.output_linear(attention_output)
        # print("input_vector:", input_vecs.size())
        # Residual 
        residual_output = input_vecs + attention_output
        residual_output = self.norm1(residual_output)
        # Feedforward
        feedforward_output = self.linear2(self.relu(self.linear1(residual_output)))
        
        # Final 
        output = feedforward_output + residual_output
        output = self.norm2(output)
        return output


    

# Part 1 Transformer train function
def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    model = Transformer(vocab_size=len(vocab_index),num_positions=500, d_model=256, d_internal=1024, num_classes=27, num_layers=1)
    # print(len(vocab_index))
    epochs = 10
    batch_size = 128
    lr = 0.01 # perplexity:10-11
    chunk_size = 20
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train(True)
    loss_func = nn.NLLLoss()

    for epoch in range(epochs):
    # train phase
        running_loss = 0.0
        for i in range(0, len(train_text), chunk_size*batch_size):
            train_x_batch = []
            train_y_batch = []
            for j in range(i, i+chunk_size*batch_size, chunk_size):
                if j+chunk_size < len(train_text)-1:
                    text_batch = train_text[j: j+chunk_size-1]
                    label_batch = train_text[j+1: j+chunk_size]
                    x_batch = [vocab_index.index_of(k) for k in text_batch]
                    y_batch = [vocab_index.index_of(k) for k in label_batch]
                    train_x_batch.append(x_batch)
                    train_y_batch.append(y_batch)
            optimizer.zero_grad()
            outputs = model(torch.tensor(train_x_batch).long())
            outputs = torch.transpose(outputs, 1, 2)
            loss = loss_func(outputs, torch.tensor(train_y_batch))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss/(i+1)
        print(f"loss: {epoch_loss}")
    # Initialize the language model
    # print(len(vocab_index))
    # model = Transformer(vocab_size=len(vocab_index),num_positions=512, d_model=200, d_internal=64, num_classes=27, num_layers=1)
    
    # # Define loss function and optimizer
    # loss_fc = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # batch_size = 16
    # # seq_len = 20  # chunk??
    # num_batches = len(train_text) // batch_size 

    # num_epochs = 1

    # for epoch in range(num_epochs):
    #     model.train()
    #     total_loss = 0.0

    #     for batch_idx in range(num_batches-1):
    #         start_idx = batch_idx * batch_size 
    #         end_idx = (batch_idx + 1) * batch_size 

    #         # batch_data
    #         batch_data = train_text[start_idx:end_idx]
    #         indices = [vocab_index.index_of(char) for char in batch_data]
    #         batch_tensor = torch.tensor(indices, dtype=torch.long)
    #         batch_tensor = batch_tensor.view(batch_size, -1)

    #         # target 
    #         target_char = train_text[start_idx + 1 : end_idx + 1] 
    #         target_indices = torch.tensor([vocab_index.index_of(char) for char in target_char], dtype=torch.long)
            
    #         # model
    #         print("batch_tensor:",batch_tensor.size())
    #         output,_ = model(batch_tensor)
    #         # print(output.size())
    #         output_flat = output.view(-1, 27)
    #         # output_flat = output.view(-1, 27)
    #         # print(log_probs.size()) # 32, 20, 27
    #         # print(output_flat.size(), target_indices.size())
            
    #         loss = loss_fc(output_flat, target_indices)

    #         # Backward pass and optimization
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         total_loss += loss.item()

    #     avg_loss = total_loss / num_batches
    #     print(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}")
    return NeuralLanguageModel(model, vocab_index)
