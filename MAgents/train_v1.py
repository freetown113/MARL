import magent2 as magent
import math
import numpy as np
from collections import deque
from scipy.spatial.distance import cityblock
import torch
from torch import optim
import copy
from random import shuffle

from model import NetworkLSTM, weights_init
from agent import DQNAgent, EpsilonGreedyActionSelector, EpsilonTracker
from erb import PrioReplayBufferNaive, BetaTracker, experience


map_size = 30
env = magent.GridWorld("battle", map_size=map_size)
env.set_render_dir("render")
team1, team2 = env.get_handles()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

beta_start = 0.3
beta_steps = 2e6
update_target = 1000
lr = 1e-4
batch_size = 32
hid_layer = 25
act_space = env.action_space[0][0]
observation_space = (13, 13, 2)  # env.view_space[0]
width = height = map_size
n1 = n2 = 16
gap = 1
epochs = 10000
replay_size = 1000
temperature = (0.5, 0.1, 1000000)
beta = (0.4, 1, 100000)
alpha = 0.4
gamma = 0.9


side1 = int(math.sqrt(n1)) * 2
pos1 = []
for x in range(width//2 - gap - side1, width//2 - gap - side1 + side1, 2):
    for y in range((height - side1)//2, (height - side1)//2 + side1, 2):
        pos1.append([x, y, 0])

side2 = int(math.sqrt(n2)) * 2
pos2 = []
for x in range(width//2 + gap, width//2 + gap + side2, 2):
    for y in range((height - side2)//2, (height - side2)//2 + side2, 2):
        pos2.append([x, y, 0])


env.reset()
env.add_agents(team1, method="custom", pos=pos1)
env.add_agents(team2, method="custom", pos=pos2)

agents_list_1 = dict({key: value for value, key in enumerate(env.get_agent_id(team1))})
agents_list_2 = dict({key: value for value, key in enumerate(env.get_agent_id(team2))})

nets_1 = []
for _ in range(1):
# for _ in range(env.get_num(team1)):
    nets_1.append(NetworkLSTM(observation_space, act_space))
    nets_1[-1].apply(weights_init('orthogonal'))
target_nets_1 = copy.deepcopy(nets_1)
nets_2 = []
for _ in range(1):
# for _ in range(env.get_num(team2)):
    nets_2.append(NetworkLSTM(observation_space, act_space))
    nets_2[-1].apply(weights_init('orthogonal'))
target_nets_2 = copy.deepcopy(nets_2)

selector = EpsilonGreedyActionSelector(temperature[0])
eps_tracker = EpsilonTracker(selector, temperature[0], temperature[1],
                             temperature[2])

buf_1 = PrioReplayBufferNaive(replay_size, alpha, beta_start)
beta_tracker_1 = BetaTracker(buf_1, beta[0], beta[1], beta[2])

buf_2 = PrioReplayBufferNaive(replay_size, alpha, beta[0])
beta_tracker_2 = BetaTracker(buf_2, beta[0], beta[1], beta[2])


agnt_1 = DQNAgent(agents_list_1, nets_1, target_nets_1, buf_1, selector, batch_size, gamma, device=device)

optimizers_1 = [optim.Adam(net_1.parameters(), lr=lr) for net_1 in nets_1]

agnt_2 = DQNAgent(agents_list_2, nets_2, target_nets_2, buf_2, selector, batch_size, gamma, device=device)

optimizers_2 = [optim.Adam(net_2.parameters(), lr=lr) for net_2 in nets_2]


def train(agent, batch_size, replay, optimizer, hidden):
    ids = np.random.randint(low=0, high=len(replay), size=batch_size)
    exps = [replay[idx] for idx in ids]
    losses = []
    jobs = torch.stack([ex[0] for ex in exps]).detach()
    jacts = torch.stack([ex[1] for ex in exps]).detach()
    jrewards = torch.stack([ex[2] for ex in exps]).detach()
    jmeans = torch.stack([ex[3] for ex in exps]).detach()
    vs = torch.stack([ex[4] for ex in exps]).detach()
    qs = []
    for h in range(batch_size):
        state = torch.cat((jobs[h].flatten(), jmeans[h]))
        # q_vals = agent.network[0](state.detach().unsqueeze(0), hidden[0])[0]
        qs.append(agent.network[0](state.detach().unsqueeze(0), hidden)[0])
    qvals = torch.stack(qs)
    target = qvals.clone().detach().squeeze(-2)
    target[:, jacts] = jrewards + gamma * torch.max(vs, dim=1)[0]  # 20 = 20 + 20
    loss = torch.sum(torch.pow(qvals.squeeze(-2) - target.detach(), 2))
    losses.append(loss.detach().item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return np.array(losses).mean()


N1 = env.get_num(team1)
N2 = env.get_num(team2)

acts_1 = torch.randint(low=0, high=act_space, size=(N1,))
acts_2 = torch.randint(low=0, high=act_space, size=(N2,))

replay1 = deque(maxlen=replay_size)
replay2 = deque(maxlen=replay_size)

qnext1 = torch.zeros(N1)
qnext2 = torch.zeros(N2)

act_means1 = agnt_1.init_mean_field(N1, act_space)
act_means2 = agnt_2.init_mean_field(N2, act_space)

rewards1 = torch.zeros(N1)
rewards2 = torch.zeros(N2)

losses1 = []
losses2 = []


def team_step(team, agent, acts, hidden):
    obs = env.get_observation(team)
    ids = env.get_agent_id(team)
    obs_small = torch.from_numpy(obs[0][:, :, :, [1, 4]])
    agent_pos = env.get_pos(team)
    acts, act_means, qvals, hidden = agent.infer_acts(obs_small, agent_pos, acts, hidden)
    return acts, act_means, qvals, obs_small, ids, hidden


def add_to_replay(replay, obs_small, acts, rewards, act_means, qnext):
    for j in range(rewards.shape[0]):
        exp = (obs_small[j], acts[j], rewards[j], act_means[j], qnext[j])
        replay.append(exp)

    return replay


counter = -1
for i in range(epochs):
    done = False
    mean_res_1 = []
    mean_res_2 = []
    hidden_1 = [None] * env.get_num(team1)
    hidden_2 = [None] * env.get_num(team2)
    step_ct = 0
    while not done:
        counter += 1
        acts_1, act_means1, qvals1, obs_small_1, ids_1, hidden_1 = \
            team_step(team1, agnt_1, acts_1, hidden_1)
        env.set_action(team1, acts_1.detach().numpy().astype(np.int32))

        acts_2, act_means2, qvals2, obs_small_2, ids_2, hidden_2 = \
            team_step(team2, agnt_2, acts_2, hidden_2)
        env.set_action(team2, acts_2.detach().numpy().astype(np.int32))

        done = env.step()

        _, _, qnext1, _, ids_1, hidden_1 = team_step(team1, agnt_1, acts_1,
                                                     hidden_1)
        _, _, qnext2, _, ids_2, hidden_2 = team_step(team2, agnt_2, acts_2,
                                                     hidden_2)

        env.render()

        rewards1 = torch.from_numpy(env.get_reward(team1)).float()
        rewards2 = torch.from_numpy(env.get_reward(team2)).float()
        mean_res_1.append(rewards1.numpy().sum())
        mean_res_2.append(rewards2.numpy().sum())

        replay1 = add_to_replay(replay1, obs_small_1, acts_1, rewards1,
                                act_means1, qnext1)
        replay2 = add_to_replay(replay2, obs_small_2, acts_2, rewards2,
                                act_means2, qnext2)
        shuffle(replay1)
        shuffle(replay2)

        ids_1_ = list(zip(np.arange(ids_1.shape[0]), ids_1))
        ids_2_ = list(zip(np.arange(ids_2.shape[0]), ids_2))

        env.clear_dead()

        ids_1  = env.get_agent_id(team1)
        ids_2  = env.get_agent_id(team2)

        ids_1_ = [i for (i, j) in ids_1_ if j in ids_1]
        ids_2_ = [i for (i, j) in ids_2_ if j in ids_2]

        acts_1 = acts_1[ids_1_]
        acts_2 = acts_2[ids_2_]

        step_ct += 1
        if step_ct > 250:
            env.reset()
            env.add_agents(team1, method="custom", pos=pos1)
            env.add_agents(team2, method="custom", pos=pos2)
            break

        if len(replay1) > batch_size and len(replay2) > batch_size:
            loss1 = train(agnt_1, batch_size, replay1, optimizers_1[0], None)
            # loss2 = train(agnt_2, batch_size, replay2, optimizers_2[0], hidden_2)
            losses1.append(loss1)
            # losses2.append(loss2)

        eps_tracker.frame(counter)
        beta_tracker_1.frame(counter)
        beta_tracker_2.frame(counter)

    if i % 1 == 0:
        print(f'For epoch {i} Mean reward_1 is {np.mean(mean_res_1):.4f} Mean reward_2 is {np.mean(mean_res_2):.4f}')
