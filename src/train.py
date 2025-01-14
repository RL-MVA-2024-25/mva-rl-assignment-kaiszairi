from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV

import os
import random
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

class MLP(nn.Module):
    def __init__(self, state_dim, nb_neurons, nb_actions, num_layers=4):
        super(MLP, self).__init__()
        
        layers = [nn.Linear(state_dim, nb_neurons), nn.ReLU()]
        
        # Add intermediate layers dynamically
        for _ in range(num_layers):
            layers.append(nn.Linear(nb_neurons, nb_neurons))
            layers.append(nn.ReLU())
        
        # Add the output layer
        layers.append(nn.Linear(nb_neurons, nb_actions))
        
        # Use nn.Sequential to stack layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

state_dim = env.observation_space.shape[0]
nb_actions = env.action_space.n
nb_neurons = 256

model = MLP(state_dim, nb_neurons, nb_actions)

config = {
    'nb_actions': env.action_space.n,
    'learning_rate': 0.001,
    'gamma': 0.98,
    'buffer_size': 100000,
    'epsilon_min': 0.02,
    'epsilon_max': 1.,
    'epsilon_decay_period': 21000, 
    'epsilon_delay_decay': 100,
    'batch_size': 800,
    'gradient_steps': 3,
    'update_target_strategy': 'replace', # or 'ema' (tried but replace seems to work better for this model/problem)
    'update_target_freq': 400,
    'update_target_tau': 0.005,
    'criterion': torch.nn.SmoothL1Loss()
}


class ProjectAgent:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def greedy_action(self, observation):
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        best_val = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_return = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                self.greedy_action(state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()

            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)

            # next transition
            step += 1
            if done or trunc:
                val_score = evaluate_HIV(agent=self, nb_episode=1)
                if val_score > best_val:
                    best_val = val_score
                    self.save()

                episode += 1

                # Monitoring
                episode_return.append(episode_cum_reward)
                print(f"Episode: {episode:3d} | Epsilon: {epsilon:6.2f} | Batch Size: {len(self.memory):5d} | Episode Return: {episode_cum_reward:.2e} | Validation Score: {val_score:.2e}")

                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state

        return episode_return

    def act(self, observation, use_random=False):
        if use_random:
            return np.random.randint(self.nb_actions)
        else:
            return self.greedy_action(observation)

    def save(self):
        torch.save(self.model.state_dict(), f'best_model.pth')

    def load(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.load_state_dict(torch.load("best_model.pth",  map_location=device, weights_only=False))


if __name__ == "__main__":
    agent = ProjectAgent()
    agent.train(env, max_episode=500)