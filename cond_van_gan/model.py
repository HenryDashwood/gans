import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim):
        super(Generator, self).__init__()
        self.fcn = nn.Sequential(
            nn.Linear(in_features=noise_dim + label_dim, out_features=240, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=240, out_features=240, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=240, out_features=240, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=240, out_features=784, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, batch, labels):
        inputs = batch.view(batch.size(0), -1)
        ret = torch.cat((inputs, labels), dim=1)
        ret = self.fcn(ret)
        return ret


class Maxout(nn.Module):
    def __init__(self, num_pieces):
        super(Maxout, self).__init__()
        self.num_pieces = num_pieces

    def forward(self, x):
        assert x.shape[1] % self.num_pieces == 0
        ret = x.view(
            *x.shape[:1], x.shape[1] // self.num_pieces, self.num_pieces, *x.shape[2:]
        )
        ret, _ = ret.max(dim=2)
        return ret


class Discriminator(nn.Module):
    def __init__(self, label_dim):
        super(Discriminator, self).__init__()
        self.fcn = nn.Sequential(
            nn.Linear(in_features=784 + label_dim, out_features=240),
            Maxout(5),
            nn.Dropout(0.5),
            nn.Linear(in_features=48, out_features=240, bias=True),
            Maxout(5),
            nn.Dropout(0.5),
            nn.Linear(in_features=48, out_features=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, batch, labels):
        ret = batch.view(batch.size(0), -1)
        ret = torch.cat((ret, labels), dim=1)
        ret = self.fcn(ret)
        return ret