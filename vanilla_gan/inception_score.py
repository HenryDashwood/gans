import math

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import inception_v3

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

net = inception_v3(pretrained=True).to(device)


def inception_score(images, batch_size):
    scores = []
    num_steps = int(math.ceil(float(len(images)) / float(batch_size)))
    for i in range(num_steps):
        s = i * batch_size
        e = (i + 1) * batch_size
        mini_batch = images[s:e]
        batch = Variable(mini_batch)
        s, _ = net(batch)
        scores.append(s)
    scores = torch.cat(scores, 0)
    p_yx = F.softmax(scores, 1)
    p_y = p_yx.mean(0).unsqueeze(0).expand(p_yx.size(0), -1)
    KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))
    final_score = KL_d.mean()
    return final_score
