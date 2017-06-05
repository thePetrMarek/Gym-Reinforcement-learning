class Debug:
    def __init__(self):
        self.history_length = 100
        self.history = []

    def add_episode_reward(self, episode_reward):
        self.history.append(episode_reward)
        if len(self.history) > self.history_length:
            self.history.pop(0)

    def show_debug(self, episode):
        self.show_actual_reward(episode)
        self.show_average_reward()

    def show_actual_reward(self, episode):
        last_reward = self.history[len(self.history) - 1]
        print(str(episode) + ". Reward: " + str(last_reward))

    def show_average_reward(self):
        print("Average: " + str(sum(self.history) / len(self.history)))
