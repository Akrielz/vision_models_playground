import torch


class TiedEmbedding(torch.nn.Module):

    def __init__(self, vocab_size: int, embedding_size: int):
        super(TiedEmbeddingSoftmax, self).__init__()

        self.w = torch.nn.Parameter(torch.randn(vocab_size, embedding_size))
        self.b = torch.nn.Parameter(torch.randn(vocab_size))

    def forward(self, inputs: torch.Tensor, embed: bool = True):
        if embed:
            return torch.nn.functional.embedding(inputs, self.w)

        return torch.tensordot(inputs, self.w.t(), 1) + self.b


def main():
    vocab_size = 1000

    for embedding_size in range(1, 50):

        tied_embedding_softmax = TiedEmbedding(vocab_size, embedding_size)
        tied_embedding_softmax.eval()

        acc = 0

        for t in range(100):
            inputs = torch.tensor([i for i in range(vocab_size)])
            embedded = tied_embedding_softmax(inputs, embed=True)

            outputs_softmax = tied_embedding_softmax(embedded, embed=False)
            outputs = torch.argmax(outputs_softmax, dim=1)

            # check if the outputs are the same as the inputs
            correct = torch.eq(inputs, outputs).sum() / len(inputs)
            acc = acc + correct.item()

        print(f"embedding_size: {embedding_size}, acc: {acc}")


if __name__ == "__main__":
    main()
