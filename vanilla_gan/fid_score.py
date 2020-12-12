from math import ceil

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models import inception_v3


class PartialInceptionNetwork(nn.Module):
    def __init__(self):
        super(PartialInceptionNetwork, self).__init__()
        self.inception_network = inception_v3(pretrained=True)
        self.inception_network.Mixed_7c.register_foward_hook(self.output_hook)

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
