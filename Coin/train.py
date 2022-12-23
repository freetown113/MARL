import agent
import environment
import erb
from erb import experience
import model
from model import weights_init

import copy
from itertools import count
import numpy as np
import time
import torch
import torch.optim as optim


NUM_AGENTS = 2
BATCH_SIZE = 32
HRS_COUNT = 5

EPS_START = 1.0
EPS_FINAL = 0.1
EPS_STEPS = 1000000

GAMMA = 0.99

ALPHA = 0.4
BETA_START = 0.4
BETA_FINAL = 1.0
BETA_STEPS = 1000000
REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
REWARD_STEPS = 2
LEARNING_RATE = 0.0001
STATES_TO_EVALUATE = 1000
UPDATE_TARGET = 1000
CHECKPOINT = 'checkpoint'


config__ = {
    "beta_start": 0.3,
    "beta_steps": 2e6,
    "update_target": 2000,
    "lr": 1e-4,
    "batch_size": 32
}


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = environment.CoinGameVec(100, 1, 5, display=False)

    net_1 = model.NetworkLSTM(env.observation_space.shape,
                                        env.action_space.n).to(device)
    net_1.apply(weights_init('orthogonal'))
    target_net_1 = copy.deepcopy(net_1)

    net_2 = model.NetworkLSTM(env.observation_space.shape,
                                        env.action_space.n).to(device)
    net_2.apply(weights_init('orthogonal'))
    target_net_2 = copy.deepcopy(net_2)

    selector = agent.EpsilonGreedyActionSelector(EPS_START)
    eps_tracker = agent.EpsilonTracker(selector, EPS_START, EPS_FINAL, 
                                       EPS_STEPS)

    buf = erb.PrioReplayBufferNaive(REPLAY_SIZE, ALPHA, config__['beta_start'])

    beta_tracker = erb.BetaTracker(buf, config__['beta_start'], BETA_FINAL, config__['beta_steps'])

    agnt_1 = agent.DQNAgent(net_1, target_net_1, buf, selector, config__['batch_size'], GAMMA, device=device)

    optimizer_1 = optim.Adam(net_1.parameters(), lr=config__['lr'])

    agnt_2 = agent.DQNAgent(net_2, target_net_2, buf, selector, config__['batch_size'], GAMMA, device=device)

    optimizer_2 = optim.Adam(net_2.parameters(), lr=config__['lr'])

    mean_loss_1 = []
    mean_rews_1 = []
    mean_loss_2 = []
    mean_rews_2 = []
    episode_reward_1 = []
    episode_reward_2 = []
    context = dict()

    state = env.reset()
    hidden = None
    start = time.process_time()
    for i in count():
        if i % config__['update_target'] == 0:
            target_net_1.load_state_dict(net_1.state_dict())
            target_net_2.load_state_dict(net_2.state_dict())
        # if i % 10000 == 0:
        #     net_2.load_state_dict(net_1.state_dict())

        state = state.to(device)
        h = hidden
        action_1, hidden = agnt_1.sample_action(state, h)
        action_2, _ = agnt_2.sample_action(state, h)
        actions = tuple(((action_1, action_2),))

        next_state, reward, done, trunc, info = env.step(actions)

        mean_rews_1.append(reward[0].numpy())
        mean_rews_2.append(reward[1].numpy())

        buf.add_element(experience(state=state.squeeze(0), action=actions,
                                   reward=reward,
                                   done=torch.tensor((1 if done else 0,),
                                                     dtype=torch.long),
                                   next_state=next_state.squeeze(0),
                                   hx=hidden[0], hc=hidden[1]))

        if done:
            context = {key: context.get(key, 0) + info.get(key, 0)
                       for key in set(context) | set(info)}
            state = env.reset()
            hidden = None
            episode_reward_1.append(np.sum(mean_rews_1))
            episode_reward_2.append(np.sum(mean_rews_2))
            mean_rews_1 = []
            mean_rews_2 = []
        else:
            state = next_state

        if len(buf) >= REPLAY_INITIAL:
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            states, act_1, act_2, rew_1, rew_2, dones, n_states, hid, indices, weights = agnt_1.get_batch_data()
            loss_1, priorities_1, indices_1 = agnt_1.train(states, act_1, rew_1, dones, n_states, hid, indices, weights)
            loss_2, priorities_2, indices_2 = agnt_2.train(states, act_2, rew_2, dones, n_states, hid, indices, weights)

            loss_1.backward()
            loss_2.backward()

            torch.nn.utils.clip_grad_norm_(net_1.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(net_2.parameters(), max_norm=0.5)
            optimizer_1.step()
            optimizer_2.step()

            # buf.update_priorities(indices_1, priorities_1)
            mean_loss_1.append(loss_1.detach())
            buf.update_priorities(indices_2, priorities_2)
            mean_loss_2.append(loss_2.detach())

        eps_tracker.frame(i)
        beta_tracker.frame(i)

        if i % 10000 == 0 and i > 0:
            mark = time.process_time()
            mean_rew_1 = np.mean(episode_reward_1)
            mean_loss_1 = np.mean(mean_loss_1)
            mean_rew_2 = np.mean(episode_reward_2)
            mean_loss_2 = np.mean(mean_loss_2)
            print(f'In iteration {i} mEreward_1 is {mean_rew_1:.3f},'
                  f' mEreward_2 is {mean_rew_2:.3f}, beta is {buf.beta:.3f}'
                  f' mloss_1 is {mean_loss_1:.7f} mloss_2 is {mean_loss_2:.7f} time taken {(mark - start):.2f}')
            #print(context)
            context = dict()
            episode_reward_1 = []
            mean_loss_1 = []
            episode_reward_2 = []
            mean_loss_2 = []



if __name__ == '__main__':
    train()