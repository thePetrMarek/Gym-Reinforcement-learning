import random

from agent import Agent
import numpy as np


class DeepQAgent(Agent):
    def __init__(self, action_space, epsilon, epsilon_min, gama, memory_size, batch_size, episodes, neural_network):
        super().__init__(action_space)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.gama = gama
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
        self.neural_network = neural_network
        self.epsilon_decay = (epsilon - epsilon_min) / episodes

    def load(self):
        self.neural_network.load()

    def add_to_memory(self, old_observation, action, reward, new_observation, done):
        self.memory.append((old_observation, action, reward, new_observation, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def get_batch_from_memory(self):
        n = min(self.batch_size, len(self.memory))
        return random.sample(self.memory, n)

    def select_action(self, observation):
        action = self.epsilon_greedy(observation)
        return action

    def epsilon_greedy(self, observation):
        if random.random() < self.epsilon:
            action = self.action_space.sample()
        else:
            observation = observation.reshape(1, len(observation))
            actions_q, action = self.neural_network.predict(observation)
            action = action[0]
        return action

    def update(self, old_observation, action, reward, new_observation, done):
        self.add_to_memory(old_observation, action, reward, new_observation, done)
        batch = self.get_batch_from_memory()

        old_observations = []
        new_observations = []
        for sample in batch:
            old_observation = sample[0]
            new_observation = sample[3]
            old_observations.append(old_observation)
            new_observations.append(new_observation)
        old_actions_q, _ = self.neural_network.predict(old_observations)
        new_actions_q, _ = self.neural_network.predict(new_observations)

        inputs = []
        labels = []
        for i in range(len(batch)):
            old_observation = batch[i][0]
            action = batch[i][1]
            reward = batch[i][2]
            done = batch[i][4]
            new_state_value = np.max(new_actions_q[i])
            target = old_actions_q[i]
            if not done:
                target[action] = reward + self.gama * new_state_value
            else:
                target[action] = reward

            inputs.append(old_observation)
            labels.append(target)

        self.neural_network.update(inputs, labels)

    def after_episode_update(self):
        self.epsilon -= self.epsilon_decay

    def save_checkpoint(self, episode):
        self.neural_network.save(episode)
