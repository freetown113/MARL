import gym
from gym.utils import seeding
from gym.envs.registration import EnvSpec
import numpy as np
import pygame
import random
import time
import torch
from typing import Iterable, List, Tuple


WHITE = (255, 255, 255)
GREY = (128, 128, 128)


class CoinGameVec:
    metadata = {'render.modes': ['human']}
    spec = EnvSpec("CoinGameVec-v0", entry_point='environment.CoinGameVec')
    """Vectorized Coin Game environment.
    Created according to the OpenAI Gym API. Describe two-agents grid
    environment. Agents pick up coins by moving onto the position where
    the coin is located. While every agent receives a point for picking
    up a coin of any colour, whenever an picks up a coin of different
    colour, the other agent loses 2 points.
    """
    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    Actions = [
        np.array([0,  1]),
        np.array([0, -1]),
        np.array([1,  0]),
        np.array([-1, 0]),
    ]

    def __init__(self,
                 max_steps: int,
                 batch_size: int,
                 grid_size: int = 3,
                 num_coins: int = 3,
                 display: bool = False
                 ) -> None:
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.num_coins = num_coins
        self.show = display
        # The 4 channels stand for 2 players and 2 coin positions
        self.ob_space_shape = [2 + 2 * self.num_coins, grid_size, grid_size]
        self.step_count = None
        self.action_space = gym.spaces.Discrete(n=len(self.Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=self.ob_space_shape,
                                                dtype=np.float32)

        if self.show:
            self.win_size = 600
            self.img_size = self.win_size / grid_size
            self.window = pygame.display.set_mode((self.win_size,
                                                   self.win_size))
            self.sim_speed = 1
            pygame.display.set_caption("CoinGameVec-v0")

            gold = pygame.image.load("images/gold_player.png").convert()
            self.Gold = pygame.transform.scale(gold,
                                               (self.img_size, self.img_size))
            black = pygame.image.load("images/black_player.png").convert()
            self.Black = pygame.transform.scale(black,
                                                (self.img_size, self.img_size))
            mix = pygame.image.load("images/mix_player.png").convert()
            self.Mix = pygame.transform.scale(mix,
                                              (self.img_size, self.img_size))

            black_coin = pygame.image.load("images/black.jpg").convert()
            self.Black_coin = pygame.transform.scale(black_coin,
                                                     (self.img_size,
                                                      self.img_size))
            gold_coin = pygame.image.load("images/gold.jpg").convert()
            self.Gold_coin = pygame.transform.scale(gold_coin,
                                                    (self.img_size,
                                                     self.img_size))
            self.grid = None

    def reset(self):
        self.info = {'gold_coins': 0,
                     'black_coins': 0,
                     'black_self': 0,
                     'black_golds': 0,
                     'gold_self': 0,
                     'gold_blacks': 0}
        self.step_count = 0
        self.black_coin = np.random.randint(2, size=self.num_coins)
        # Agent and coin positions
        self.black_pos = np.random.randint(self.grid_size,
                                           size=(2))
        self.gold_pos = np.random.randint(self.grid_size,
                                          size=(2))
        self.coin_pos = np.zeros((self.num_coins, 2), dtype=np.int8)
        # Make sure coins don't overlap
        while self._same_pos(self.black_pos, self.gold_pos):
            self.gold_pos = np.random.randint(self.grid_size, size=2)
        for i in range(self.num_coins):
            self._generate_coin(i)

        if self.show:
            self.grid = make_grid(self.grid_size, self.win_size)
            self.grid[self.black_pos.item(0)][self.black_pos.item(1)] \
                .type = self.Black
            self.grid[self.gold_pos.item(0)][self.gold_pos.item(1)].type \
                = self.Gold
            draw(self.window, self.grid, self.grid_size, self.win_size)
            time.sleep(self.sim_speed)
            for idx, coin in enumerate(self.coin_pos):
                if self.black_coin[idx]:
                    self.grid[coin[0]][coin[1]].type \
                        = self.Black_coin
                else:
                    self.grid[coin[0]][coin[1]].type \
                        = self.Gold_coin
                draw(self.window, self.grid, self.grid_size, self.win_size)
                time.sleep(self.sim_speed)

        return self._generate_state()

    def _generate_coin(self, i):
        self.black_coin[i] = 1 - self.black_coin[i]

        if self.black_coin[i]:
            self.info['black_coins'] += 1
        else:
            self.info['gold_coins'] += 1
        # Make sure coin has a different position than the agents
        success = 0
        while success < 2 + i:
            self.coin_pos[i] = np.random.randint(self.grid_size, size=(2))
            success = 1 - self._same_pos(self.black_pos,
                                         self.coin_pos[i])
            for idx in range(0, i):
                success += 1 - self._same_pos(self.coin_pos[idx],
                                              self.coin_pos[i])
            success += 1 - self._same_pos(self.gold_pos,
                                          self.coin_pos[i])

    def _same_pos(self, x, y):
        return (x == y).all()

    def _generate_state(self):
        state = np.zeros(self.ob_space_shape,
                         dtype=np.float32)
        state[0, self.black_pos[0], self.black_pos[1]] = 1
        state[1, self.gold_pos[0], self.gold_pos[1]] = 1
        for idx, coin in enumerate(self.coin_pos):
            if self.black_coin[idx]:
                state[(idx * 2) + 2, coin[0], coin[1]] = 1
            else:
                state[(idx * 2) + 3, coin[0], coin[1]] = 1
        return torch.from_numpy(state).unsqueeze(0)

    def step(self,
             actions: np.array
             ) -> Tuple[np.array, float, bool, bool, str]:
        ac0, ac1 = actions[0]
        ac0, ac1 = ac0.item(), ac1.item()
        assert ac0 in {0, 1, 2, 3} and ac1 in {0, 1, 2, 3}

        if self.show:
            self.grid[self.black_pos.item(0)][self.black_pos.item(1)] \
                .type = None
            self.grid[self.gold_pos.item(0)][self.gold_pos.item(1)].type \
                = None

        self.black_pos = \
            (self.black_pos + self.Actions[ac0]) % self.grid_size
        self.gold_pos = \
            (self.gold_pos + self.Actions[ac1]) % self.grid_size

        if self.show:
            if self._same_pos(self.black_pos, self.gold_pos):
                self.grid[self.black_pos.item(0)][self.black_pos.item(1)] \
                    .type = self.Mix
            else:
                self.grid[self.black_pos.item(0)][self.black_pos.item(1)] \
                    .type = self.Black
                self.grid[self.gold_pos.item(0)][self.gold_pos.item(1)] \
                    .type = self.Gold
            draw(self.window, self.grid, self.grid_size, self.win_size)
            time.sleep(self.sim_speed)

        # Compute rewards
        reward_black, reward_gold = [], []
        indices = []
        for idx, coin in enumerate(self.coin_pos):
            # generate = False
            if self.black_coin[idx]:
                if self._same_pos(self.gold_pos, coin):
                    # generate = True
                    reward_black.append(0)
                    reward_gold.append(0)
                    # self.info['gold_blacks'] += 1
                    indices.append(idx)
                elif self._same_pos(self.black_pos, coin):
                    # generate = True
                    reward_black.append(1)
                    reward_gold.append(-1)
                    # self.info['black_self'] += 1
                    indices.append(idx)
                else:
                    reward_black.append(0)
                    reward_gold.append(0)

            else:
                if self._same_pos(self.gold_pos, coin):
                    # generate = True
                    reward_black.append(0)
                    reward_gold.append(0)
                    indices.append(idx)
                elif self._same_pos(self.black_pos, coin):
                    # generate = True
                    reward_black.append(2)
                    reward_gold.append(-2)
                    # self.info['black_golds'] += 1
                    indices.append(idx)
                else:
                    reward_black.append(0)
                    reward_gold.append(0)
        for idx in indices[::-1]:
            self.coin_pos = np.delete(self.coin_pos, idx, axis=0)
            self.black_coin = np.delete(self.black_coin, idx)

            # if generate:
            #     self._generate_coin(i)
        
        if self.show:
            for idx, coin in enumerate(self.coin_pos):
                if self.black_coin[idx]:
                    self.grid[coin[0]][coin[1]] \
                        .type = self.Black_coin
                else:
                    self.grid[coin[0]][coin[1]] \
                        .type = self.Gold_coin
                draw(self.window, self.grid, self.grid_size, self.win_size)
                time.sleep(self.sim_speed)

        reward = [torch.tensor(reward_black).sum(), torch.tensor(reward_gold).sum()]
        self.step_count += 1
        done = np.array(self.step_count == self.max_steps or self.coin_pos.size == 0)
        state = self._generate_state()
        trunc = True
        info = self.info  # 'some information'

        return state, reward, done, trunc, info

    def render(self):
        pass

    def close(self):
        pass

    def seed(self,
             seed: int = None
             ) -> List:
        self.np_random, seed1 = seeding.np_random(seed)
        self.np_random, seed2 = seeding.np_random(hash(seed1 + 1) % 2 ** 31)
        return [seed1, seed2]

    @classmethod
    def get_env(cls, **kwargs):
        return CoinGameVec(**kwargs)


class Cell:
    def __init__(self,
                 row: int,
                 col: int,
                 width: int,
                 ) -> None:
        self.x = row * width
        self.y = col * width
        self.width = width
        self.color = WHITE
        self.type = None

    def get_pos(self):
        return self.x, self.y

    def draw(self,
             win: pygame.display
             ) -> None:
        pygame.draw.rect(win, self.color,
                         (self.x, self.y, self.width, self.width))


def iter_sample_fast(iterable: Iterable,
                     samplesize: int
                     ) -> List:
    results = []
    try:
        for _ in range(samplesize):
            results.append(next(iterable))
    except StopIteration:
        raise ValueError("Sample larger than population.")
    random.shuffle(results)  # Randomize their positions
    for i, v in enumerate(iterable, samplesize):
        r = random.randint(0, i)
        if r < samplesize:
            results[r] = v  # at a decreasing rate, replace random items
    return results


def make_grid(rows: int,
              width: int
              ) -> List:
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            cell = Cell(i, j, gap)
            grid[i].append(cell)

    return grid


def draw_grid(win: pygame.display,
              rows: int,
              width: int
              ) -> None:
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win: pygame.display,
         grid: int,
         rows: int,
         width: int
         ) -> None:
    win.fill(WHITE)

    for row in grid:
        for cell in row:
            cell.draw(win)
            if cell.type is not None:
                win.blit(cell.type, (cell.x, cell.y))

    draw_grid(win, rows, width)
    pygame.display.update()
