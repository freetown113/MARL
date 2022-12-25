import numpy as np
import torch
from typing import Any, Callable, Dict, Tuple


class EpsilonGreedyActionSelector:
    def __init__(self,
                 epsilon: float = 0.05,
                 selector: Callable = None
                 ) -> None:
        self.epsilon = epsilon
        self.selector = selector if selector is not None else np.argmax

    def __call__(self,
                 scores: Tuple | np.array
                 ) -> np.array:
        match scores:
            case tuple():
                assert scores[0].shape == scores[1].shape
                batch_size, n_actions = scores[0].shape
                actions = []
                for s in scores:
                    action = self.selector(s, axis=1)
                    mask = np.random.random(size=batch_size) < self.epsilon
                    rand_actions = np.random.choice(n_actions, sum(mask))
                    action[mask] = rand_actions
                    actions.append(action)
                return torch.tensor(actions).reshape(batch_size, -1, 1)
            case np.ndarray():
                batch_size, n_actions = scores.shape
                actions = self.selector(scores, axis=1)
                mask = np.random.random(size=batch_size) < self.epsilon
                rand_actions = np.random.choice(n_actions, sum(mask))
                actions[mask] = rand_actions
                return actions


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
                      agent: str,
                      states: Tuple | np.array,
                      hidden: torch.Tensor
                      ) -> np.array:
        if torch.is_tensor(states):
            states = states.to(self.device)
        q_v, hidden = self.network[self.agents_list[agent]](states, hidden)
        match q_v:
            case tuple():
                q = (q_v[0].numpy(), q_v[1].numpy())
            case torch.Tensor():
                q = q_v.numpy()
        actions = self.action_selector(q)
        return torch.from_numpy(actions), hidden

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
