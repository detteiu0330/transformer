import torch


def masking(dot_product):
    masked = []
    mask = [0] * dot_product.size(1)
    for i in range(dot_product.size(0)+1):
        sequence = [dot_product[j].tolist() if i > j else mask for j in range(dot_product.size(0))]
        masked.append(sequence)
    masked = torch.Tensor(masked)
    return masked


def main():
    dot_product = torch.randn((7, 4))
    masked = masking(dot_product)


if __name__ == '__main__':
    main()


