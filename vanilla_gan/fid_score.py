from math import ceil

import numpy as np
from scipy import linalg
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models import inception_v3


class PartialInceptionNetwork(nn.Module):
    def __init__(self):
        super(PartialInceptionNetwork, self).__init__()
        self.inception_network = inception_v3(pretrained=True)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)

    def output_hook(self, module, input, output):
        self.mixed_7c_output = output

    def forward(self, x):
        assert x.shape[1:] == (
            3,
            299,
            299,
        ), f"Expected input shape (N, 3, 299, 299) but got {x.shape}"
        self.inception_network(x)
        activations = self.mixed_7c_output
        activations = F.adaptive_avg_pool2d(activations, (1, 1))
        activations = activations.view(x.shape[0], 2048)
        return activations


def fid_score(real_images, gen_images, batch_size, device):
    net = PartialInceptionNetwork().to(device)
    real_activations = []
    gen_activations = []
    num_steps = int(ceil(float(len(real_images)) / float(batch_size)))
    for i in range(num_steps):
        s = i * batch_size
        e = (i + 1) * batch_size
        batch_real = Variable(real_images[s:e])
        batch_gen = Variable(gen_images[s:e])
        features_real = net(batch_real)
        features_gen = net(batch_gen)
        real_activations.append(features_real)
        gen_activations.append(features_gen)
    features_real = torch.cat(real_activations, 0)
    features_gen = torch.cat(gen_activations, 0)
    xr = features_real.detach().cpu().numpy()
    xg = features_gen.detach().cpu().numpy()
    u1 = np.mean(xr, axis=0)
    u2 = np.mean(xg, axis=0)
    s1 = np.cov(xr, rowvar=False)
    s2 = np.cov(xg, rowvar=False)
    diff = u1 - u2
    diff_squared = diff.dot(diff)
    prod = s1.dot(s2)
    sqrt_prod, _ = linalg.sqrtm(prod, disp=False)
    if np.iscomplexobj(sqrt_prod):
        sqrt_prod = sqrt_prod.real
    prod_tr = np.trace(sqrt_prod)
    final_score = diff_squared + np.trace(s1) + np.trace(s2) - 2 * prod_tr
    return final_score