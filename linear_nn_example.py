import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        x = self.linear(x)

        return x


def run():
    torch.manual_seed(1024)

    model = Model()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)

    w1 = 1.233

    w2 = 3.6

    b = 0.988
    num_iteration = 5000
    for i in range(num_iteration):
        optimizer.zero_grad()
        x1 = np.random.random()
        x2 = np.random.random()
        y = w1*x1+w2*x2 +b
        y = torch.tensor([[y]])
        pred = model.forward(torch.tensor([[x1, x2]]))


        loss = F.mse_loss(y, pred)
        loss.backward()
        optimizer.step()

    for name, param in model.named_parameters():
        print(name)
        print(param)


if __name__ == "__main__":
    run()