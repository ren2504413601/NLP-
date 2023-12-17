'''
基于 torch 实现的 transformer
可以对照原文来看，基本结构像MuiltHeadAttention、PositionWiseFeedForward
以及 positional_encoding 都有实现
结合 Attention Is All You Need 原论文比对来看更加方便
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        
        # Reshape to separate heads
        query = query.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # Apply attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        
        # Reshape and linear transformation
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        attention_output = self.output_linear(attention_output)
        
        return attention_output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size, ff_size):
        super(PositionWiseFeedForward, self).__init__()
        self.hidden_size = hidden_size
        self.ff_size = ff_size
        
        self.linear1 = nn.Linear(hidden_size, ff_size)
        self.linear2 = nn.Linear(ff_size, hidden_size)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class Transformer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_size, num_layers):
        super(Transformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.positional_encoding = self._create_positional_encoding(hidden_size)
        
        self.encoder = nn.ModuleList([self._create_encoder_layer() for _ in range(num_layers)])
        self.decoder = nn.ModuleList([self._create_decoder_layer() for _ in range(num_layers)])
        
        self.output_linear = nn.Linear(hidden_size, output_vocab_size)
        
    def _create_positional_encoding(self, hidden_size):
        max_seq_len = 100  # Set your maximum sequence length
        positional_encoding = torch.zeros(max_seq_len, hidden_size)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding.unsqueeze(0)
    
    def _create_encoder_layer(self):
        attention = MultiHeadAttention(self.hidden_size, self.num_heads)
        feed_forward = PositionWiseFeedForward(self.hidden_size, self.ff_size)
        return nn.ModuleList([attention, feed_forward])
    
    def _create_decoder_layer(self):
        self_attention = MultiHeadAttention(self.hidden_size, self.num_heads)
        encoder_attention = MultiHeadAttention(self.hidden_size, self.num_heads)
        feed_forward = PositionWiseFeedForward(self.hidden_size, self.ff_size)
        return nn.ModuleList([self_attention, encoder_attention, feed_forward])
    
    def forward(self, src_input, tgt_input):
        src_mask = (src_input != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt_input != 0).unsqueeze(1).unsqueeze(2)
        
        src_embedded = self.embedding(src_input) + self.positional_encoding[:, :src_input.size(1)]
        tgt_embedded = self.embedding(tgt_input) + self.positional_encoding[:, :tgt_input.size(1)]
        
        src_output = src_embedded
        tgt_output = tgt_embedded
        
        for i in range(self.num_layers):
            # Encoder layer
            attention = self.encoder[i][0]
            feed_forward = self.encoder[i][1]
            
            src_output = src_output + attention(src_output, src_output, src_output, src_mask)
            src_output = src_output + feed_forward(src_output)
            
            # Decoder layer
            self_attention = self.decoder[i][0]
            encoder_attention = self.decoder[i][1]
            feed_forward = self.decoder[i][2]
            
            tgt_output = tgt_output + self_attention(tgt_output, tgt_output, tgt_output, tgt_mask)
            tgt_output = tgt_output + encoder_attention(tgt_output, src_output, src_output, src_mask)
            tgt_output = tgt_output + feed_forward(tgt_output)
        
        output = self.output_linear(tgt_output)
        return output