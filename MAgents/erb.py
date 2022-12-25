from collections import namedtuple
import numpy as np
import random
import torch
from typing import List, NamedTuple, Tuple

experience = namedtuple('Experience', 'state, action, reward, done, \
                         next_state, hx, hc')


class ExperienceReplay:
    '''Simple Experience Replay (ER) buffer, that contains all elements in the
    list. Elements sempled from the ER with uniform probability distribulion.
    Elements inside ER are stored in namedtuples.
    '''
    def __init__(self,
                 capacity: int,
                 alpha: float = 0.4
                 ) -> None:
        self.capacity = capacity
        self.buffer = []
        self.pointer = 0

    def __len__(self):
        return len(self.buffer)

    def sample(self,
               batch: int
               ) -> List:
        if self.__len__() < batch:
            raise RuntimeError('There are not enough elements to sample from \
            the buffer')
        examples = random.sample(self.buffer, batch)
        return examples

    def add_element(self,
                    element: NamedTuple
                    ) -> None:
        if self.__len__() < self.capacity:
            self.buffer.append(element)
        else:
            self.buffer[self.pointer % self.capacity] = element
            self.pointer += 1


class PrioReplayBufferNaive:
    '''Prioritize Experience Replay realized in a simple way (not time
    efficient). All elements are stored as namedtuples and sampled with
    the probability according to their priority. Initial priority of a sample
    added is maximum priority. Priority is updated after the element is
    sampled. New priority is based on the algorithm's error for this sample.
    '''
    def __init__(self,
                 capacity: int,
                 prob_alpha: float,
                 beta: float
                 ) -> None:
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.beta = beta
        self.pointer = 0
        self.buffer = []
        self.priorities = np.zeros((capacity, ), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def add_element(self,
                    element: NamedTuple
                    ) -> None:
        max_prio = self.priorities.max() if self.buffer else 1.0

        if self.__len__() < self.capacity:
            self.buffer.append(element)
        else:
            self.buffer[self.pointer % self.capacity] = element
        self.pointer += 1
        self.priorities[self.pointer % self.capacity] = max_prio

    def sample(self,
               batch: int
               ) -> Tuple:
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pointer]
        probs = np.array(prios, dtype=np.float32) ** self.prob_alpha

        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch, p=probs,
                                   replace=True)
        samples = [self.buffer[idx] for idx in indices]
        total = self.__len__()
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self,
                          batch_indices: np.array,
                          batch_priorities: np.array
                          ) -> None:
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def show_priorities(self) -> None:
        for i in range(self.__len__()):
            print(f'index {i} priority {self.priorities[i]}')


class PrioReplayBuffer:
    '''Prioritize Experience Replay realized in a time efficient way. The
    underlaying structure is a tree. Two different trees are used. One for
    calculating min element and another to get sum of the elements in the
    tree. Such finding min element and the sum is much faster.
    '''
    def __init__(self,
                 capacity: int,
                 alpha: float,
                 beta: float
                 ) -> None:
        self.buffer = []
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
        self.pointer = 0

        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2

        self.min_tree = [float('inf') for i in range(2 * self.capacity)]
        self.sum_tree = [0 for i in range(2 * self.capacity)]

    def __len__(self):
        return len(self.buffer)

    def priority_set(self,
                     index: int,
                     priority: float,
                     name: str
                     ) -> None:
        index %= self.capacity
        index += self.capacity
        match name:
            case 'min':
                tree = self.min_tree
            case 'sum':
                tree = self.sum_tree

        tree[index] = priority

        while index >= 2:
            index //= 2
            match name:
                case 'min':
                    tree[index] = min(tree[index * 2], tree[index * 2 + 1])
                case 'sum':
                    tree[index] = tree[index * 2] + tree[index * 2 + 1]

    def add_element(self,
                    element: NamedTuple
                    ) -> None:
        if self.__len__() < self.capacity:
            self.buffer.append(element)
        else:
            self.buffer[self.pointer % self.capacity] = element

        self.priority_set(self.pointer, self.max_priority ** self.alpha, 'min')
        self.priority_set(self.pointer, self.max_priority ** self.alpha, 'sum')
        self.pointer += 1

    def find_prefix_sum_index(self,
                              prefix_sum: float
                              ) -> int:
        idx = 1
        while idx < self.capacity:
            if self.sum_tree[idx * 2] > prefix_sum:
                idx *= 2
            else:
                prefix_sum -= self.sum_tree[idx * 2]
                idx = idx * 2 + 1

        return idx - self.capacity

    def sample(self,
               batch: int
               ) -> Tuple:
        weights, indices = [], []
        if self.beta <= 0:
            raise ValueError(f'Beta should be grater than 0, '
                             f'provided {self.beta}')

        for i in range(batch):
            mass = random.random() * self.sum_tree[1]
            index = self.find_prefix_sum_index(mass)
            indices.append(index)

        prob_min = self.min_tree[1] / self.sum_tree[1]
        max_wght = (prob_min * self.__len__()) ** (-self.beta)

        for i in range(batch):
            prob = self.sum_tree[indices[i] + self.capacity] / self.sum_tree[1]
            weight = (prob * self.__len__()) ** (-self.beta)
            weights.append(weight / max_wght)

        assert not np.array(indices).max() == np.array(indices).min()

        weights = np.array(weights, dtype=np.float32)
        samples = [self.buffer[idx] for idx in indices]

        return samples, indices, weights

    def update_priorities(self,
                          indices: np.array,
                          priorities: np.array
                          ) -> None:
        for idx, prior in zip(indices, priorities):
            self.max_priority = max(self.max_priority, prior)

            priority = prior ** self.alpha

            self.priority_set(idx, priority, 'min')
            self.priority_set(idx, priority, 'sum')

    def show_priorities(self) -> None:
        for i in range(self.__len__()):
            print(f'index {i} priority {self.min_tree[i]} '
                  f'{self.min_tree[i + self.capacity]}')


class BetaTracker:
    """
    Updates beta parameter according to linear schedule
    """
    def __init__(self, selector: PrioReplayBuffer,
                 beta_start: int | float,
                 beta_final: int | float,
                 beta_frames: int):
        self.selector = selector
        self.beta_start = beta_start
        self.beta_final = beta_final
        self.beta_frames = beta_frames
        self.frame(0)

    def frame(self,
              frame: int
              ) -> None:
        beta = self.beta_start + frame / self.beta_frames
        self.selector.beta = min(self.beta_final, beta)


if __name__ == '__main__':
    buffer = PrioReplayBuffer(50, 0.4)

    for i in range(75):
        state = torch.randn((1, 84, 84, 3), dtype=torch.float32)
        action = torch.randn((1,), dtype=torch.float32)
        reward = torch.randn((1,), dtype=torch.float32)
        done = torch.randn((1,), dtype=torch.float32)
        exp = experience(state=state, action=action, reward=reward, done=done)
        buffer.add_element(exp)
        print(f'size: {len(buffer)}')

    print(f'Buffer size: {len(buffer)}')

    for i in range(75):
        state = torch.randn((1, 84, 84, 3), dtype=torch.float32)
        action = torch.randn((1,), dtype=torch.float32)
        reward = torch.randn((1,), dtype=torch.float32)
        done = torch.randn((1,), dtype=torch.float32)
        exp = experience(state=state, action=action, reward=reward, done=done)
        buffer.add_element(exp)

        el, indices, weights = buffer.sample(2, 0.6)

    buffer.show_priorities()
    for i in range(10):
        buffer.update_priorities([1, 2, 3, 4, 5, 7],
                                 [1.0, 0.9, 0.3, 0.2, 1.0, 0.06])
    buffer.show_priorities()
