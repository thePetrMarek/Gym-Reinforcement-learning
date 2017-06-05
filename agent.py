class Agent:
    def __init__(self, action_space):
        self.action_space = action_space
        pass

    def load(self):
        pass

    def select_action(self, observation):
        pass

    def update(self, old_observation, action, reward, new_observation, done):
        pass

    def after_episode_update(self):
        pass

    def save_checkpoint(self, episode):
        pass
