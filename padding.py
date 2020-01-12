import torch

pad = torch.LongTensor([3])


def padding(inputs, output_length):
    outputs = [inputs[i].tolist() if i < inputs.size(0) else pad for i in range(output_length)]
    outputs = torch.LongTensor(outputs)
    return outputs


def main():
    inputs = torch.randint(0, 10, (7, ))
    print(inputs)
    outputs = padding(inputs, 20)
    print(outputs)


if __name__ == '__main__':
    main()
