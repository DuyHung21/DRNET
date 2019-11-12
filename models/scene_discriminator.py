import torch
import torch.nn as nn
import torch.nn.functional as F

class SceneDiscriminator(nn.Module):
    def __init__(self, d_in, d_hidden=100):
        super(SceneDiscriminator, self).__init__()

        self.d_in = d_in
        self.layers = nn.Sequential(
            nn.Linear(d_in * 2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, pose1, pose2):
        inp = torch.cat([pose1, pose2], 1)
        inp = inp.view(pose1.shape[0], -1)
        return self.layers(inp)


if __name__ == '__main__':
    inp = torch.randn(4,3,8,8)
    inp2 = torch.randn(4,3,8,8)
    net = SceneDiscriminator(3 * 8 * 8)
    out = net(inp, inp2)

    print(out.shape)

    