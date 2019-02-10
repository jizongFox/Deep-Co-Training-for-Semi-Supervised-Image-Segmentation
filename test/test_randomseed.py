import torch
import random, torch, os, numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


class Model(torch.nn.Module):
    def __init__(self, input_shape, num_classes=10):
        super(Model, self).__init__()

        # if channel dim not present, add 1
        if len(input_shape) == 2:
            input_shape.append(1)
        H, W, C = input_shape

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(C, 64, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(64, 192, kernel_size=3, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(192, 192, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.H_out = H // (2 * 2 * 2)
        self.W_out = W // (2 * 2 * 2)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(192 * self.H_out * self.W_out, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 192 * self.H_out * self.W_out)
        x = self.classifier(x)
        return x


def main():
    seed_everything()
    batch_size = 1024
    dataset = CIFAR10('./data',
                      download=True,
                      train=True,
                      transform=ToTensor()
                      )
    model = Model((32, 32, 3)).cuda()
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    for epoch in range(3):
        print(f'epoch {epoch}')
        print(f'Kernel weight sum: {model.features[0].weight.sum()}')
        data_iter = iter(dataloader)
        while True:
            try:
                X, Y = next(data_iter)
                data, labels = X.cuda(async=True), Y.cuda(async=True)
                optimizer.zero_grad()
                output = model(data)
                loss = CrossEntropyLoss()(output, labels)
                loss.backward()
                optimizer.step()
            except StopIteration:
                break


if __name__ == '__main__':
    main()
