import tensorflow as tf


class NeuralNetwork:
    def __init__(self, save_name, load_name):
        self.session = tf.Session()
        self.save_name = save_name
        self.load_name = load_name

    def update(self, batch, labels):
        pass

    def predict(self, input):
        pass

    def save(self, episode):
        pass

    def load(self):
        pass
