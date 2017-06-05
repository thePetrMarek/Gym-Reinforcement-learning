import gym
from gym import wrappers

from DeepQAgent.deep_q_agent import DeepQAgent
from DeepQAgent.deep_q_neural_network import DeepQNeuralNetwork
from debug import Debug


class Main:
    def __init__(self):
        self.api_key = 'YOUR_API_KEY'
        self.environment = 'CartPole-v0'
        self.env = gym.make(self.environment)
        self.episodes = 3000
        self.agent_name = "DeepQAgent"
        self.neural_network = DeepQNeuralNetwork("./DeepQAgent/" + self.environment + "/model", "")
        self.agent = DeepQAgent(self.env.action_space, 1.0, 0.1, 0.90, 50, 50, self.episodes, self.neural_network)
        self.load = False
        self.train = True
        self.render = True
        self.monitor = True
        self.upload = False
        self.checkpoint_every = 100

    def run(self):
        debug = Debug()
        if self.monitor:
            self.env = wrappers.Monitor(self.env, './DeepQAgent/' + self.environment, video_callable=False, force=True)
        if self.load:
            self.agent.load()

        for episode in range(self.episodes):
            step = 0
            total_reward = 0
            observation = self.env.reset()

            while True:
                if self.render:
                    self.env.render()
                action = self.agent.select_action(observation)
                new_observation, reward, done, info = self.env.step(action)
                total_reward += reward
                if self.train:
                    self.agent.update(observation, action, reward, new_observation, done)
                observation = new_observation
                if done:
                    break
                step += 1
            self.agent.after_episode_update()

            if episode % self.checkpoint_every == 0:
                self.agent.save_checkpoint(episode)
            debug.add_episode_reward(total_reward)
            debug.show_debug(episode)

        self.agent.save_checkpoint(self.episodes)
        if self.upload:
            gym.upload('/tmp/' + self.environment, api_key=self.api_key)


if __name__ == "__main__":
    main = Main()
    main.run()
