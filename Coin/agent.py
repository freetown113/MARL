import numpy as np
import torch
from typing import Any, Callable, Tuple


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
        self.batch_size = batch_size
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device
        self.gamma = gamma
        self.criterion = torch.nn.HuberLoss(reduction='none')

    def initial_state(self):
        """
        Should create initial empty state for the self. It will be called for
        the start of the episode
        :return: Anything self want to remember
        """
        return None

    @torch.no_grad()
    def sample_action(self,
                      states: Tuple | np.array
                      ) -> np.array:
        if torch.is_tensor(states):
            states = states.to(self.device)
        q_v = self.network(states)
        match q_v:
            case tuple():
                q = (q_v[0].numpy(), q_v[1].numpy())
            case torch.Tensor():
                q = q_v.numpy()
        actions = self.action_selector(q)
        return torch.from_numpy(actions)

    def train(self,
              states: torch.Tensor,
              actions: Tuple,
              rewards: Tuple,
              dones: Tuple,
              n_states: torch.Tensor,
              indices: Tuple,
              weights: np.array
              ) -> Tuple:

        q_vals = self.network(states)
        q_pred = q_vals.gather(-1, actions)

        with torch.no_grad():
            next_q_vals = self.network(n_states)
            target_q_vals = self.target_network(n_states)

        next_act = torch.argmax(next_q_vals, dim=-1)
        next_q_pred = target_q_vals.gather(-1, next_act.unsqueeze(-1)) \
            .squeeze(-1)

        Q = rewards + self.gamma * next_q_pred.unsqueeze(-1) * (1 - dones)

        td_error = (Q - q_pred).detach()
        priorities = td_error.abs().numpy()

        losses = self.criterion(Q, q_pred)
        loss = (losses * torch.from_numpy(weights).unsqueeze(-1)).sum()

        return loss, priorities, indices

    def get_batch_data(self) -> Tuple:
        batch_data, indices, weights = self.buffer.sample(self.batch_size)

        elements = zip(*batch_data)

        states = torch.stack(next(elements))
        actions = next(elements)
        act_1, act_2 = torch.chunk(torch.tensor(actions), 2, dim=-1)
        act_1 = act_1.reshape(self.batch_size, -1)
        act_2 = act_2.reshape(self.batch_size, -1)
        rewards = next(elements)
        rew_1, rew_2 = torch.chunk(torch.tensor(rewards), 2, dim=-2)
        rew_1 = rew_1.reshape(self.batch_size, -1)
        rew_2 = rew_2.reshape(self.batch_size, -1)
        dones = torch.stack(next(elements))
        n_states = torch.stack(next(elements))

        return states, act_1, act_2, rew_1, rew_2, dones, n_states, indices, \
            weights


class DistributionalDQNAgent(DQNAgent):
    def __init__(self,
                 network: Callable,
                 target_network: Callable,
                 buf: Any,
                 action_selector: Callable,
                 batch_size: int,
                 gamma: float = 0.99,
                 device: str = 'cpu',
                 preprocessor: Callable = default_states_preprocessor
                 ) -> None:
        super().__init__(network, target_network, buf,
                         action_selector, batch_size,
                         gamma, device, preprocessor)
        self.n_atoms = 51
        self.V_min = -20
        self.V_max = 20

    @torch.no_grad()
    def sample_action(self,
                      states: torch.Tensor
                      ) -> Tuple:
        q_value_dist = self.network(states)
        q_value_dist = q_value_dist * torch.linspace(self.V_min, self.V_max,
                                                     self.n_atoms)

        # actions = torch.argmax(q_value_dist.sum(-1), axis=-1)
        actions = torch.from_numpy(self.action_selector(q_value_dist.sum(-1).numpy()))
        return actions

    def train(self,
              states: torch.Tensor,
              actions: Tuple,
              rewards: Tuple,
              dones: Tuple,
              n_states: torch.Tensor,
              indices: Tuple,
              weights: np.array
              ) -> Tuple:
        q_value_dist = self.network(states)
        actions = actions.unsqueeze(-1).expand(self.batch_size, 1,
                                               self.n_atoms)
        q_values = q_value_dist.gather(-2, actions)
        q_values = q_values.squeeze(-2)
        # q_values = q_values.clamp(0.01, 0.99)

        delta_z = float(self.V_max - self.V_min) / (self.n_atoms - 1)
        support = torch.linspace(self.V_min, self.V_max, self.n_atoms)

        next_q_value_dist = self.target_network(n_states) * support
        next_actions = torch.argmax(next_q_value_dist.sum(-1), axis=-1) \
            .unsqueeze(-1).unsqueeze(-1)

        next_actions = next_actions.expand(self.batch_size, 1, self.n_atoms)
        next_q_values = next_q_value_dist.gather(-2, next_actions).squeeze(-2)

        rewards = rewards.expand_as(next_q_values)
        dones = dones.expand_as(next_q_values)

        Tz = rewards + (1 - dones) * 0.99 * support
        Tz = Tz.clamp(min=self.V_min, max=self.V_max)
        bz = (Tz - self.V_min) / delta_z
        low = bz.floor().long()
        upp = bz.ceil().long()

        offset = torch.linspace(0, (self.batch_size - 1) * self.n_atoms,
                                self.batch_size).long() \
            .view(self.batch_size, 1).expand(self.batch_size, self.n_atoms)

        target_q_values = torch.zeros(next_q_values.size())
        target_q_values.view(-1).index_add_(0, (low + offset).view(-1),
                                            (next_q_values *
                                            (upp.float() - bz)).view(-1),)
        target_q_values.view(-1).index_add_(0, (upp + offset).view(-1),
                                            (next_q_values *
                                            (bz - low.float())).view(-1),)

        loss = -(target_q_values * q_values.log()).sum(1)

        priorities = loss.detach().abs().numpy()

        loss = (loss * torch.from_numpy(weights)).mean()
        assert not np.isnan(priorities).any()
        return loss, priorities, indices
