import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.multiprocessing import Process
import gym
import numpy as np
from torch.autograd import Variable
from torch.optim import Adam
import os
import matplotlib.pyplot as plt


class Agent(nn.Module):
    def __init__(self, input_dims=4, action_dims=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, 256)
        self.policy = nn.Linear(256, action_dims)
        self.value = nn.Linear(256, 1)
        self.saved_actions = []
        self.rewards = []
        self.gamma = 0.9

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.policy(x)), self.value(x)

    def select_action(self, state):
        prob, value = self.forward(state)
        action = prob.multinomial().data
        prob = prob.gather(1, Variable(action))
        self.saved_actions.append((prob.log(), value))
        return action


def get_loss(model):
    R = 0
    saved_actions = model.saved_actions
    value_loss = 0
    policy_loss = 0
    rewards = []
    for r in model.rewards[::-1]:
        R = r + model.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.data[0,0]
        policy_loss += -(log_prob*reward)
        value_loss += F.smooth_l1_loss(value, Variable(torch.Tensor([r])))
    del model.rewards[:]
    del model.saved_actions[:]
    return value_loss + policy_loss


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def average_rewards(reward):
    size = float(dist.get_world_size())
    dist.all_reduce(reward, op=dist.reduce_op.SUM)
    reward /= size
    return reward


def train(rank, size):
    agent = Agent()
    optimizer = Adam(agent.parameters(), lr=0.001)
    rewards = []

    for episode in np.arange(100):
        state = env.reset()
        for step in np.arange(2000):
            state = torch.from_numpy(state).float().unsqueeze(0)
            action = agent.select_action(Variable(state))
            next_state, reward, done, _ = env.step(action[0,0])
            agent.rewards.append(reward)
            if done:
                break
            state = next_state

        avg_reward = average_rewards(torch.ones(1, 1)*float(step))
        rewards.append(avg_reward.numpy())

        optimizer.zero_grad()
        loss = get_loss(agent)
        loss.backward()
        average_gradients(agent)
        optimizer.step()
        #print('Rank ', dist.get_rank(), ', epoch ',
        #      episode, ': ', loss)

    if rank == 0:
        np.savetxt('rewards.csv', rewards, delimiter=',')


def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, train))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    rewards = np.genfromtxt('rewards.csv', delimiter=',')
    plt.plot(rewards)