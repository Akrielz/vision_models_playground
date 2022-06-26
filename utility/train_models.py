import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


def train_model(
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        optimizer: torch.optim.Optimizer = None,
        num_epochs: int = 100,
        batch_size: int = 64
):
    # get train loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # get test loader
    test_loader = None
    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # init optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train model
    for epoch in range(num_epochs):
        for data, target in tqdm(train_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = F.cross_entropy(output, target)

            # backward
            loss.backward()

            # update
            optimizer.step()
            optimizer.zero_grad()

        # test model
        if test_loader is not None:
            with torch.no_grad():
                correct = 0
                total = 0
                for data, target in test_loader:
                    data, target = data.cuda(), target.cuda()
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += len(target)

        print('Test Accuracy: %f' % (correct / total))
