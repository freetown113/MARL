import math
import numpy as np
import torch
import torch.nn as nn


def weights_init(init_type: str = 'xavier'):
    '''Function that set neural network's parameters according to provided with
    the argument logic. Destinated to use fot NN parameters initialization.

    input: string
    possible values ['normal', 'xavier', 'kaiming', 'orthogonal']

    Example:
    network: nn.Module
    network.apply(weights_init('xavier'))  # initialize network's parametrs
    '''
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) \
           and hasattr(m, 'weight'):
            if init_type == 'normal':
                torch.nn.init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    return init_fun


class Network(nn.Module):
    '''Neural network architecture build according to the Dueling DQN algorithm
    Takes tensor representing states from the environment as input and returns
    Q-values for each possible action from this state. Initially output is
    divided into value function V(s) and andantage A(s,a). Their sum represents
    Q(s,a) = V(s) + A(s,a). THis permits lerning process to be more stable.
    '''
    def __init__(self,
                 shape: torch.Size,
                 n_actions: int,
                 n_agents: int
                 ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(shape[0], 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_agents)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def _get_conv_out(self,
                      shape: torch.Size
                      ) -> int:
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class DistributionalNetwork(nn.Module):
    '''Neural network architecture that permits build the Distributional DQN
    algorithm. It returns a distribution for the Q-value instead of it's mean.
    The whole space between min and max values is divided by 51 atom.
    '''
    def __init__(self,
                 input_shape: torch.Size,
                 n_actions: int,
                 n_agent: int
                 ) -> None:
        super().__init__()
        self.n_atoms = 51
        self.Vmin = -20
        self.Vmax = 20
        self.Delta_z = (self.Vmax - self.Vmin) / (self.n_atoms - 1)

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(out_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions * self.n_atoms)
        )

        sups = torch.arange(self.Vmin, self.Vmax + self.Delta_z, self.Delta_z)
        self.register_buffer("supports", sups)
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self,
                      shape: torch.Size
                      ) -> int:
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        batch_size = x.shape[0]
        conv_out = self.conv(x).view(batch_size, -1)
        fc_out = self.fc(conv_out)
        return self.softmax(fc_out.view(-1, self.n_atoms)) \
            .view(batch_size, -1, self.n_atoms)
