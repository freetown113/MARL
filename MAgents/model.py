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


class NetworkLSTM(nn.Module):
    '''Neural network architecture build according to the Dueling DQN algorithm
    Takes tensor representing states from the environment as input and returns
    Q-values for each possible action from this state. Initially output is
    divided into value function V(s) and andantage A(s,a). Their sum represents
    Q(s,a) = V(s) + A(s,a). THis permits lerning process to be more stable.
    '''
    def __init__(self,
                 shape: torch.Size,
                 n_actions: int,
                 hidden: int = 32,
                 n_agents: int = 1,
                 is_training: bool = True
                 ) -> None:
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')
        self.in_size = np.product(shape)
        self.shape = shape
        self.hidden = hidden
        self.training = is_training
        self.conv = nn.Sequential(
            nn.Linear(self.in_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.fc_val = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

        self.lstm = nn.LSTM(32, hidden, batch_first=False)

    def _init_hidden(self, size) -> torch.Tensor:
        if self.training is True:
            return torch.zeros([1, size, self.hidden]).to(self.device), \
                   torch.zeros([1, size, self.hidden]).to(self.device)
        else:
            return torch.zeros([1, 1, self.hidden]).to(self.device), \
                   torch.zeros([1, 1, self.hidden]).to(self.device)

    def forward(self,
                x: torch.Tensor,
                hidden: torch.Tensor
                ) -> torch.Tensor:
        conv_out = self.conv(x).view(1, x.size()[0], -1)
        if hidden is None:
            hidden = self._init_hidden(x.shape[0])
        lstm, hidden = self.lstm(conv_out, hidden)
        lstm = lstm.squeeze(-2)
        val = self.fc_val(lstm)
        adv = self.fc_adv(lstm)
        return val + (adv - adv.mean(dim=1, keepdim=True)), hidden
