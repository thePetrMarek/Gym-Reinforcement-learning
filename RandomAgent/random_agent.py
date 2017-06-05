from agent import Agent


class RandomAgent(Agent):
    def select_action(self, observation):
        return self.action_space.sample()

    def update(self, old_observation, action, reward, new_observation, done):
        pass

    def save_checkpoint(self, episode):
        pass
