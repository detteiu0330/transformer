import torch.nn as N

from positional_encoding import positional_encoding
from MHA import MultiHeadAttention
from FFN import FeedForward

PAD = 3


class Encoder(N.Module):
    def __init__(self, vocab_size, output_size):
        super(Encoder, self).__init__()
        self.embedding_layer = N.Embedding(vocab_size + 4, output_size, padding_idx=PAD)
        self.self_attention = MultiHeadAttention(output_size, output_size, output_size, depth_size=2)
        self.layer_norm1 = N.LayerNorm(output_size)
        self.feed_forward = FeedForward(output_size)
        self.layer_norm2 = N.LayerNorm(output_size)

    def forward(self, inputs):
        inputs = self.embedding_layer(inputs)
        inputs = positional_encoding(inputs)
        attention = self.self_attention(inputs, inputs, inputs)
        attention = attention + inputs
        attention = self.layer_norm1(attention)
        outputs = self.feed_forward(attention)
        outputs = attention + outputs
        outputs = self.layer_norm2(outputs)
        print('encode is over')
        return outputs
