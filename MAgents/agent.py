import numpy as np
import torch
from typing import Any, Callable, Dict, Tuple
from scipy.spatial.distance import cityblock


class EpsilonGreedyActionSelector:
    def __init__(self,
                 temperature: float = 0.5,
                 selector: Callable = None
                 ) -> None:
        self.temperature = temperature
        self.selector = selector if selector is not None else torch.multinomial

    def __call__(self,
                 scores: torch.Tensor
                 ) -> torch.Tensor:
        soft = torch.exp(scores / self.temperature) / \
               torch.sum(torch.exp(scores / self.temperature))
        action = self.selector(soft, 1)
        return action


class EpsilonTracker:
    """
    Updates epsilon according to linear schedule
    """
    def __init__(self, selector: EpsilonGreedyActionSelector,
                 eps_start: int | float,
                 eps_final: int | float,
                 eps_frames: int):
        self.selector = selector
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_frames = eps_frames
        self.frame(0)

    def frame(self,
              frame: int
              ) -> None:
        eps = self.eps_start - frame / self.eps_frames
        self.selector.epsilon = max(self.eps_final, eps)


def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we
    assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states],
                             copy=False)
    return torch.tensor(np_states)


class DQNAgent:
    """
    DQNself is a memoryless DQN self which calculates Q values
    from the observations and  converts them into the actions using
    action_selector
    """
    def __init__(self,
                 agents_list: Dict,
                 network: Callable,
                 target_network: Callable,
                 buf: Any,
                 action_selector: Callable,
                 batch_size: int,
                 gamma: float = 0.99,
                 device: str = 'cpu',
                 preprocessor: Callable = default_states_preprocessor
                 ) -> None:
        self.network = network
        self.target_network = target_network
        self.buffer = buf
        self.agents_list = agents_list
        self.batch_size = batch_size
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device
        self.gamma = gamma
        self.criterion = torch.nn.HuberLoss(reduction='none')

    @torch.no_grad()
    def sample_action(self,
                      agent: int,
                      q_v: torch.Tensor
                      ) -> torch.Tensor:
        actions = self.action_selector(q_v)
        return actions

    def train(self,
              agent: str,
              states: torch.Tensor,
              actions: Tuple,
              rewards: Tuple,
              dones: Tuple,
              n_states: torch.Tensor,
              hid_states: torch.Tensor,
              indices: Tuple,
              weights: np.array
              ) -> Tuple:

        q_vals, hidden = self.network[self.agents_list[agent]](states, hid_states)
        q_pred = q_vals.squeeze(0).gather(-1, actions)

        with torch.no_grad():
            next_q_vals, _ = self.network[self.agents_list[agent]](n_states, hidden)
            target_q_vals, _ = self.target_network[self.agents_list[agent]](n_states, hidden)

        next_act = torch.argmax(next_q_vals, dim=-1)
        next_q_pred = target_q_vals.gather(-1, next_act.unsqueeze(-1)) \
            .squeeze(-1).squeeze(0)

        Q = rewards.unsqueeze(-1) + self.gamma * next_q_pred.unsqueeze(-1) * (1 - dones)

        td_error = (Q - q_pred).detach()
        priorities = td_error.abs().numpy()

        losses = self.criterion(Q, q_pred)
        loss = (losses * torch.from_numpy(weights).unsqueeze(-1)).sum()

        return loss, priorities, indices

    def get_batch_data(self) -> Tuple:
        batch_data, indices, weights = self.buffer.sample(self.batch_size)

        elements = zip(*batch_data)

        states = torch.stack(next(elements))
        actions = torch.stack(next(elements))
        rewards = torch.stack(next(elements))
        dones = torch.stack(next(elements))
        n_states = torch.stack(next(elements))
        hx = torch.stack(next(elements)).reshape(1, self.batch_size, -1)
        hc = torch.stack(next(elements)).reshape(1, self.batch_size, -1)

        return states, actions, rewards, dones, n_states, (hx, hc), \
            indices, weights

    def get_neighbors(self, j, pos_list, r=6):
        neighbors = []
        pos_j = pos_list[j]
        for i, pos in enumerate(pos_list):
            if i == j:
                continue
            dist = cityblock(pos, pos_j)
            if dist < r:
                neighbors.append(i)
        return neighbors

    def get_onehot(self, a, act_space=21):
        x = torch.zeros(act_space)
        x[a] = 1
        return x

    def get_scalar(self, v):
        return torch.argmax(v)

    def get_mean_field(self, j, pos_list, act_list, r=7, acts=21):
        neighbors = self.get_neighbors(j, pos_list, r=r)
        mean_field = torch.zeros(acts)
        for k in neighbors:
            act_ = act_list[k]
            act = self.get_onehot(act_)
            mean_field += act
        tot = mean_field.sum()
        mean_field = mean_field / tot if tot > 0 else mean_field
        return mean_field

    def infer_acts(self, obs, pos_list, acts, hidden, act_space=21, num_iter=5):
        N = acts.shape[0]
        mean_fields = torch.zeros(N, act_space)
        acts_ = acts.clone()
        qvals = torch.zeros(N, act_space)
        h = hidden
        for i in range(num_iter):
            for j in range(N):
                mean_fields[j] = self.get_mean_field(j, pos_list, acts_)

            for j in range(N):
                state = torch.cat((obs[j].flatten(), mean_fields[j]))
                qs, h[j] = self.network[0](state.unsqueeze(0), h[j])
                qvals[j, :] = qs[:]
                acts_[j] = self.sample_action(j, qs.detach())
        return acts_, mean_fields, qvals, h

    def init_mean_field(self, N, act_space=21):
        mean_fields = torch.abs(torch.rand(N, act_space))
        for i in range(mean_fields.shape[0]):
            mean_fields[i] = mean_fields[i] / mean_fields[i].sum()
        return mean_fields
