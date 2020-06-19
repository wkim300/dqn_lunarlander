import gym
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pickle
import os
from utils import *
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random
import numpy as np

class Agent():
    """
    Agent interacts & learns from the environment
    """
    def __init__(self, state_size, action_size,  alpha, gamma=gamma, batch_size=batch_size,
                 eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, update_freq=update_freq,
                 layer1_size=layer1_size, layer2_size=layer2_size, seed=0):
        """
        Initializes the agent.
        :param state_size: dim. state
        :param action_size: dim. action
        :param alpha: learning rate
        :param eps_start: epsilon value at the beginning of training
        :param eps_end: epsilon terminal
        :param eps_decay: eps_start will decay to eps_end linearly over this no. of iterations
        :param seed: random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.eps_decay_slope = (self.eps_start-self.eps_end)/self.eps_decay  # decay slope
        self.gamma = gamma

        # initialize some stuff
        self.dqn = DQN(state_size, action_size, layer1_size, layer2_size, seed).to(device)
        self.dqn_target = DQN(state_size, action_size, layer1_size, layer2_size).to(device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())  # initialize with the same weights
        self.dqn_target.eval()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=alpha)
        self.memory = ReplayMemory(batch_size = batch_size, action_size=action_size)

        self.iter = 0  # number of iterations done
        self.steps = 0  # total number of steps taken;initialize to zero
        self.update_freq = update_freq  # update the target dqn model

    def select_action(self, state):
        """
        Given state, act randomly with probability epsilon
        :param state:
        :return: action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # convert state back to tensor & assign it to device
        self.eps = max(self.eps_end, self.eps_start - self.eps_decay_slope * self.iter) # linearly decreases eps & after eps_decay, remains at eps_end
        self.iter += 1

        # if value greater than epsilon, take the optimal action
        if random.random() > self.eps:
            with torch.no_grad():
                action_vals = self.dqn(state)

            return action_vals.argmax().item()
        # else, return random action
        else:
            return np.random.randint(0, 4)

    def learn(self, state, action, next_state, reward, done):
        """
        Given an experience tutple, train DQN
        :param state:
        :param action:
        :param next_state:
        :param reward:
        :param done:
        :return:
        """
        self.memory.add(state, action, next_state, reward, done)

        # if there is not enough experiences in the memory, don't optimize
        if len(self.memory.memory) < self.batch_size:
            return
        else:
            batch = self.memory.sample()  # get sampled batch of torch.tensor(s)
            state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch

            # Compute Q(s_t, a, weight_t) by first computing Q(s_t) through DQN(s_t)
            # and applying the actions taken from that state
            Q_approx = self.dqn(state_batch).gather(1, action_batch.long())  # .gather(1, inds) allows for column selection in all rows according to inds

            # Get max Q(s_t+1, a) from the target model(w/ frozen weights)
            Q_max_next = self.dqn_target(next_state_batch).max(1)[0].unsqueeze(1).detach()
            Q_target = reward_batch + (1-done_batch) * (self.gamma * Q_max_next)

            # Minimize the MSE
            loss = F.mse_loss(Q_target, Q_approx)  # compute the mean-squared-error(MSE)
            self.optimizer.zero_grad()  # zero out grads
            loss.backward()
            #for param in self.dqn.parameters():
                #param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            # Update for the bootstrap method
            if self.steps % self.update_freq == 0:
                self.dqn_target.load_state_dict(self.dqn.state_dict())


class ReplayMemory(object):
    '''Memory buffer for replay. For experience addition & random sampling'''
    def __init__(self, action_size, batch_size, memory_size=int(1e5), seed=0):
        self.memory_size = memory_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.exp = namedtuple("experience", field_names=["state", "action", "next_state", "reward", "done"])  # transition/experience tuple

    def add(self, state, action, next_state, reward, done):
        """Adds observed experience to the replay buffer"""
        self.memory.append(self.exp(state, action, next_state, reward, done))

    def sample(self):
        """
        Sample a batch of experience from ReplayMemory
        :return: batches of experience (torch.tensor)
        """
        exp = random.sample(self.memory, k=self.batch_size)
        exp = self.exp(*zip(*exp))  # separate into different batches (tuples)

        state_batch = torch.from_numpy(np.vstack([s for s in exp.state])).float().to(device)
        action_batch = torch.from_numpy(np.vstack([a for a in exp.action])).float().to(device)
        next_state_batch = torch.from_numpy(np.vstack([s for s in exp.next_state])).float().to(device)
        reward_batch = torch.from_numpy(np.vstack([r for r in exp.reward])).float().to(device)
        done_batch = torch.from_numpy(np.vstack([d for d in exp.done]).astype(int)).float().to(device)

        return state_batch, action_batch, next_state_batch, reward_batch, done_batch

class DQN(nn.Module):
    '''Neural network for approximating Q function'''
    def __init__(self, state_size, action_size, layer1_size, layer2_size, seed=0):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, layer1_size)
        self.fc2 = nn.Linear(layer1_size, layer2_size)
        self.fc3 = nn.Linear(layer2_size, action_size)

    def forward(self, state):
        '''
        Performs forward pass through QNN
        Returns: action values (for an input state)
        '''
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        return self.fc3(out)


# Custom function for training DQN
def train_dqn(agent, n_episodes=2000, early_termination=True):
    env = gym.make('LunarLander-v2')
    env.seed(0)
    n_episodes = n_episodes
    rewards = []                        # rewards history
    reward_window = deque(maxlen=100)  # for averaging purposes

    # For number of episodes
    for episode in range(1, n_episodes+1):
        state = env.reset()
        total_reward = 0

        # Interact & learn until simulation is over
        for t in count():
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
#             print(next_state)
            agent.learn(state, action, next_state, reward, done)
            state = next_state
            total_reward += reward
            if done:
                break
        reward_window.append(total_reward)
        rewards.append(total_reward)


        print('\rEp {}\tepsilon: {:.2f}\tAvg. reward: {:.2f}'.format(episode,  agent.eps, np.mean(reward_window)), end="")
        if episode % 100 == 0:
            print('\rEp {}\tepsilon: {:.2f}\tAvg. reward: {:.2f}'.format(episode,  agent.eps, np.mean(reward_window)))

        if early_termination:
            if np.mean(reward_window)>=200.0:
                print('\nSolved in {:d} episodes!\tAvg Score: {:.2f}'.format(episode, np.mean(reward_window)))
                print('')
                torch.save(agent.dqn.state_dict(), 'dqn_alpha{}_gamma{}_epsilon{}.pth'.format(alpha, gamma, eps_decay))
                break

    return rewards, agent

# Experiment function
def experiment(alphas, gammas,eps_decays, eps_starts, eps_ends, early_termination=True):
    train_total_rewards_history = []  # list of all total rewards observed during training
    test_total_rewards_history = []  # for storing the rewards per trial for 100 tirals using the trained agent

    for alpha, gamma, eps_decay, eps_start, eps_end in zip(alphas, gammas, eps_decays, eps_starts, eps_ends):

        # Make folder if the folder does not exist
        newpath = r'./alpha{}_gamma{}_epsilon{}'.format(alpha, gamma, eps_decay)
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        # initialize the agent
        agent = Agent(state_size=n_states, action_size=n_actions, alpha=alpha, gamma=gamma, batch_size=batch_size,
                      eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, update_freq=update_freq,
                      layer1_size=layer1_size, layer2_size=layer2_size, seed=0)

        # train the agent
        print('')
        print('Training for alpha={}, gamma={}, epsilon={}'.format(alpha, gamma, eps_decay))
        train_total_reward, agent = train_dqn(agent, n_episodes=1500, early_termination=early_termination)
        train_total_rewards_history.append(train_total_reward)

        # Save the trained agent
        with cd(newpath):
            torch.save(agent.dqn.state_dict(), 'dqn_alpha{}_gamma{}_epsilon{}.pth'.format(alpha, gamma, eps_decay))

        # Simulate for 100 episodes and store record
        test_total_rewards = []
        for ep in range(100):
            env.seed(0)
            state = env.reset()
            test_total_reward = 0

            # Interact & learn until simulation is over
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.learn(state, action, next_state, reward, done)
                state = next_state
                test_total_reward += reward
                if done:
                    test_total_rewards.append(test_total_reward)
                    break

        test_total_rewards_history.append(test_total_rewards) # Save the simulation history

        # Save the results
        with cd(newpath):
            # Save the total rewards observed during training processes
            with open('train_rewards_alpha{}_gamma{}_epsilon{}.obj'.format(alpha, gamma, eps_decay), 'wb') as fp:
                pickle.dump(train_total_reward, fp)

    # Save history of train rewards
    with open('train_rewards_history.obj', 'wb') as fp:
        pickle.dump(train_total_rewards_history, fp)

    # Save the reward per trial for 100 trials using your trained agent
    with open('test_rewards.obj', 'wb') as fp:
        pickle.dump(test_total_rewards_history, fp)

    return train_total_rewards_history, test_total_rewards_history

# Main loop for testing
if __name__ == "__main__":
    # Set Lunar Lander Environment
    env = gym.make('LunarLander-v2')
    env.seed(0)
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    # if GPU resource exitst, use it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyperparameters (fixed)
    batch_size=32
    gamma=0.9
    eps_start=1.0
    eps_end=0.1
    eps_decay=50000
    update_freq = 10
    layer1_size=64
    layer2_size=64

    # Perform grid search
    alphas= [5e-3, 1e-2] #[1e-5, 1e-4, 2e-4, 5e-4, 1e-3]
    gammas= [0.99, 0.99] #[0.99, 0.99, 0.99, 0.99, 0.99]
    eps_decays= [1, 1] #[100000, 100000, 100000, 100000, 100000]
    experiment(alphas, gammas, eps_decays, eps_starts=[1.0], eps_ends=[0.05])
