import math

import torch
from torch import nn as N

from mask import masking


def scaled_dot_product_attention(q, k, v, depth):
    a = torch.matmul(q, torch.t(k))
    a = a / math.sqrt(depth)
    a = torch.softmax(a, dim=1)
    a = torch.matmul(a, v)
    return a


def mask_scaled_dot_product_attention(q, k, v, depth, device):
    a = torch.matmul(q, torch.t(k))
    a = a / math.sqrt(depth)
    a = torch.softmax(a, dim=1)
    a = masking(a)
    a = a.to(device)
    a = torch.matmul(a, v)
    return a


class MultiHeadAttention(N.Module):
    def __init__(self, input_size, hidden_size, output_size, depth_size):
        super(MultiHeadAttention, self).__init__()
        # make each matrix
        self.v_linear = N.Linear(input_size, hidden_size)
        self.k_linear = N.Linear(input_size, hidden_size)
        self.q_linear = N.Linear(input_size, hidden_size)
        # make query heads
        self.q_head1 = N.Linear(hidden_size, hidden_size // depth_size)
        self.q_head2 = N.Linear(hidden_size, hidden_size // depth_size)
        # make key heads
        self.k_head1 = N.Linear(hidden_size, hidden_size // depth_size)
        self.k_head2 = N.Linear(hidden_size, hidden_size // depth_size)
        # make value heads
        self.v_head1 = N.Linear(hidden_size, hidden_size // depth_size)
        self.v_head2 = N.Linear(hidden_size, hidden_size // depth_size)
        # make the output layer
        self.output = N.Linear(hidden_size, output_size)

        # save the depth value
        self.depth = depth_size

    def forward(self, query, key, value, mask=False):
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)
        q_h1 = self.q_head1(query)
        q_h2 = self.q_head2(query)
        k_h1 = self.k_head1(key)
        k_h2 = self.k_head2(key)
        v_h1 = self.v_head1(value)
        v_h2 = self.v_head2(value)
        query_head = [q_h1, q_h2]
        key_head = [k_h1, k_h2]
        value_head = [v_h1, v_h2]
        attention = []
        for q, k, v in zip(query_head, key_head, value_head):
            attention.append(scaled_dot_product_attention(q, k, v, self.depth))
        attention = torch.cat(attention, dim=1)
        output = self.output(attention)
        return output


class Mask_MultiHeadAttention(N.Module):
    def __init__(self, input_size, hidden_size, output_size, depth_size, device):
        super(Mask_MultiHeadAttention, self).__init__()
        # make each matrix
        self.v_linear = N.Linear(input_size, hidden_size)
        self.k_linear = N.Linear(input_size, hidden_size)
        self.q_linear = N.Linear(input_size, hidden_size)
        # make query heads
        self.q_head1 = N.Linear(hidden_size, hidden_size // depth_size)
        self.q_head2 = N.Linear(hidden_size, hidden_size // depth_size)
        # make key heads
        self.k_head1 = N.Linear(hidden_size, hidden_size // depth_size)
        self.k_head2 = N.Linear(hidden_size, hidden_size // depth_size)
        # make value heads
        self.v_head1 = N.Linear(hidden_size, hidden_size // depth_size)
        self.v_head2 = N.Linear(hidden_size, hidden_size // depth_size)
        # make the output layer
        self.output = N.Linear(hidden_size, output_size)

        # save the depth value
        self.depth = depth_size
        # save the device
        self.device = device

    def forward(self, query, key, value):
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)
        q_h1 = self.q_head1(query)
        q_h2 = self.q_head2(query)
        k_h1 = self.k_head1(key)
        k_h2 = self.k_head2(key)
        v_h1 = self.v_head1(value)
        v_h2 = self.v_head2(value)
        query_head = [q_h1, q_h2]
        key_head = [k_h1, k_h2]
        value_head = [v_h1, v_h2]
        attention = []
        for q, k, v in zip(query_head, key_head, value_head):
            attention.append(mask_scaled_dot_product_attention(q, k, v, self.depth, self.device))
        attention = torch.cat(attention, dim=2)
        output = self.output(attention)
        return output

