import math


def positional_encoding(sentence):
    d_model = sentence.size(1)//2
    for pos in range(sentence.size(0)):
        for i in range(d_model):
            sentence[pos][i] += math.sin(float(pos)/math.pow(10000.0, i/d_model))
            sentence[pos][i+d_model] += math.cos(float(pos)/math.pow(10000.0, i/d_model))
    return sentence