from magent2.environments import battle_v4
import numpy as np
import torch
import torch.optim as optim
import copy

from model import NetworkLSTM, weights_init
from agent import DQNAgent, EpsilonGreedyActionSelector, EpsilonTracker
from erb import PrioReplayBufferNaive, BetaTracker, experience


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
REPLAY_SIZE = 10000
REPLAY_INITIAL = 1000
REWARD_STEPS = 2
LEARNING_RATE = 0.0001
STATES_TO_EVALUATE = 1000
UPDATE_TARGET = 1000
CHECKPOINT = 'checkpoint'


config__ = {
    "beta_start": 0.3,
    "beta_steps": 2e6,
    "update_target": 1000,
    "lr": 1e-4,
    "batch_size": 32
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = battle_v4.env(map_size=20,  render_mode=None, max_cycles=1000)
# print(env.observation_spaces, env.action_spaces)
env.reset()

team_1 = env.agents[:env.num_agents//2]
team_2 = env.agents[env.num_agents//2:]

agents_list_1 = dict({key: value for value, key in enumerate(team_1)})
agents_list_2 = dict({key: value for value, key in enumerate(team_2)})

nets_1 = []
for _ in range(env.num_agents):
    nets_1.append(NetworkLSTM(env.observation_spaces['red_0'].shape, env.action_spaces['red_0'].n))
    nets_1[-1].apply(weights_init('orthogonal'))
target_nets_1 = copy.deepcopy(nets_1)
nets_2 = []
for _ in range(env.num_agents):
    nets_2.append(NetworkLSTM(env.observation_spaces['blue_0'].shape, env.action_spaces['blue_0'].n))
    nets_2[-1].apply(weights_init('orthogonal'))
target_nets_2 = copy.deepcopy(nets_2)

selector = EpsilonGreedyActionSelector(EPS_START)
eps_tracker = EpsilonTracker(selector, EPS_START, EPS_FINAL, 
                                    EPS_STEPS)

buf_1 = PrioReplayBufferNaive(REPLAY_SIZE, ALPHA, config__['beta_start'])
beta_tracker = BetaTracker(buf_1, config__['beta_start'], BETA_FINAL, config__['beta_steps'])

buf_2 = PrioReplayBufferNaive(REPLAY_SIZE, ALPHA, config__['beta_start'])
beta_tracker = BetaTracker(buf_2, config__['beta_start'], BETA_FINAL, config__['beta_steps'])


agnt_1 = DQNAgent(agents_list_1, nets_1, target_nets_1, buf_1, selector, config__['batch_size'], GAMMA, device=device)

optimizers_1 = [optim.Adam(net_1.parameters(), lr=config__['lr']) for net_1 in nets_1]

agnt_2 = DQNAgent(agents_list_2, nets_2, target_nets_2, buf_2, selector, config__['batch_size'], GAMMA, device=device)

optimizers_2 = [optim.Adam(net_2.parameters(), lr=config__['lr']) for net_2 in nets_2]

update = config__['update_target'] * env.num_agents
hidden = None
previous = None

rews = dict({key: 0 for key in env.agents})

for idx, agent in enumerate(env.agent_iter()):
    if idx % update == 0 and idx > 0:
        [target_net_1.load_state_dict(net_1.state_dict()) for net_1, target_net_1 in zip(nets_1, target_nets_1)]
        [target_net_2.load_state_dict(net_2.state_dict()) for net_2, target_net_2 in zip(nets_2, target_nets_2)]

    observation, reward, termination, truncation, info = env.last()
    observation = torch.from_numpy(observation).reshape(-1, observation.size)

    h = hidden
    if agent.split('_')[0] == 'red':
        action, hidden = agnt_1.sample_action(agent, observation, h)
        if previous is not None:
            buf_1.add_element(experience(state=previous.squeeze(0), action=action,
                                    reward=torch.tensor(reward),
                                    done=torch.tensor((1 if termination else 0,),
                                                        dtype=torch.long),
                                    next_state=observation.squeeze(0),
                                    hx=hidden[0], hc=hidden[1]))
    elif agent.split('_')[0] == 'blue':
        action, hidden = agnt_2.sample_action(agent, observation, h)
        if previous is not None:
            buf_2.add_element(experience(state=previous.squeeze(0), action=action,
                                    reward=torch.tensor(reward),
                                    done=torch.tensor((1 if termination else 0,),
                                                        dtype=torch.long),
                                    next_state=observation.squeeze(0),
                                    hx=hidden[0], hc=hidden[1]))
    else:
        raise RuntimeError('Unknown agent type')

    rews[agent] += reward

    if len(buf_1) >= REPLAY_INITIAL and len(buf_2) >= REPLAY_INITIAL:
        [optim_1.zero_grad() for optim_1 in optimizers_1] 
        [optim_2.zero_grad() for optim_2 in optimizers_2]

        for a in team_1:
            loss_1, priorities_1, indices_1 = agnt_1.train(a, *agnt_1.get_batch_data())
            loss_1.backward()

        for a in team_2:
            loss_2, priorities_2, indices_2 = agnt_2.train(a, *agnt_2.get_batch_data())
            loss_2.backward()

        [torch.nn.utils.clip_grad_norm_(net_1.parameters(), max_norm=0.5) for net_1 in nets_1]
        [torch.nn.utils.clip_grad_norm_(net_2.parameters(), max_norm=0.5) for net_2 in nets_2]
        [optim_1.step() for optim_1 in optimizers_1]
        [optim_2.step() for optim_2 in optimizers_2]

    if not env.agents or sum(tuple(env.truncations.values())) == env.num_agents:
        print(rews)
        # print(idx, agent, env.agents)
        env.reset()
        hidden = None
        previous = None
        rews = dict({key: 0 for key in env.agents})

    if termination:
        print(f'Reward of eliminated {agent} is {rews[agent]}')
        # print(idx, agent, env.num_agents)
        action = None
    else:
        action = action.item()

    previous = observation
    env.step(action)
    # print(idx, agent, env.num_agents)
