import torch
import torch.nn as N

from MHA import MultiHeadAttention, Mask_MultiHeadAttention
from FFN import FeedForward
from positional_encoding import positional_encoding

PAD = 3


class Decoder(N.Module):
    def __init__(self, vocab_size, output_size, device):
        super(Decoder, self).__init__()
        self.embedding_layer = N.Embedding(vocab_size + 4, output_size, padding_idx=PAD)
        self.mask_attention = Mask_MultiHeadAttention(output_size, output_size, output_size, depth_size=2, device=device)
        self.norm_layer1 = N.LayerNorm(output_size)
        self.self_attention = MultiHeadAttention(output_size, output_size, output_size, depth_size=2)
        self.norm_layer2 = N.LayerNorm(output_size)
        self.feed_forward = FeedForward(output_size)
        self.norm_layer3 = N.LayerNorm(output_size)
        self.FFN1 = N.Linear(output_size*20, 4096)
        self.FFN2 = N.Linear(4096, vocab_size)

    def forward(self, inputs, encoder_outputs):
        outputs_array = []
        inputs = self.embedding_layer(inputs)
        positional_encoding(inputs)
        attentions = self.mask_attention(inputs, encoder_outputs, encoder_outputs)
        attentions = attentions
        for attention in attentions:
            attention = attention + inputs
            attention = self.norm_layer1(attention)
            self_attention = self.self_attention(attention, attention, attention)
            self_attention = self_attention + inputs
            self_attention = self.norm_layer2(self_attention)
            outputs = self.feed_forward(self_attention)
            outputs = self_attention + outputs
            outputs = torch.flatten(outputs)
            outputs = self.FFN1(outputs)
            outputs = self.FFN2(outputs)
            outputs = torch.softmax(outputs, dim=0)
            outputs_array.append(outputs)
        outputs_array = torch.stack(outputs_array, dim=0)
        print('decode is over')
        return outputs_array

