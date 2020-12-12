from torch import nn


class Generator(nn.Module):
    def __init__(self, noise_dim: int):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        self.fcn = nn.Sequential(
            # Fully Connected Layer 1
            nn.Linear(in_features=self.noise_dim, out_features=240, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Fully Connected Layer 2
            nn.Linear(in_features=240, out_features=240, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Fully Connected Layer 3
            nn.Linear(in_features=240, out_features=240, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Fully Connected Layer 4
            nn.Linear(in_features=240, out_features=784, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, batch):
        ret = batch.view(batch.size(0), -1)
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
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fcn = nn.Sequential(
            # Fully Connected Layer 1
            nn.Linear(in_features=784, out_features=240, bias=True),
            Maxout(5),
            # Fully Connected Layer 2
            nn.Linear(in_features=48, out_features=240, bias=True),
            Maxout(5),
            # Fully Connected Layer 3
            nn.Linear(in_features=48, out_features=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, batch):
        ret = batch.view(batch.size(0), -1)
        ret = self.fcn(ret)
        return ret
