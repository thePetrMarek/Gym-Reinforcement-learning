import tensorflow as tf

from neural_network import NeuralNetwork


class DeepQNeuralNetwork(NeuralNetwork):
    def __init__(self, save_name, load_name):
        super().__init__(save_name, load_name)

        self.input_vec = tf.placeholder(shape=[None, 4], dtype=tf.float32)
        self.target_q = tf.placeholder(shape=[None, 2], dtype=tf.float32)

        w1 = tf.Variable(tf.random_uniform([4, 128], 0, 0.01))
        b1 = tf.Variable(tf.zeros([128]))
        layer_one = tf.nn.relu(tf.matmul(self.input_vec, w1) + b1)

        w2 = tf.Variable(tf.random_uniform([128, 128], 0, 0.01))
        b2 = tf.Variable(tf.zeros([128]))
        layer_two = tf.nn.relu(tf.matmul(layer_one, w2) + b2)

        w3 = tf.Variable(tf.random_uniform([128, 2], 0, 0.01))
        b3 = tf.Variable(tf.zeros([2]))
        self.q = tf.matmul(layer_two, w3) + b3

        self.selected_action = tf.argmax(self.q, 1)

        loss = tf.reduce_mean(tf.square(self.q - self.target_q))
        self.train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)

        init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()
        self.session.run(init)

    def update(self, batch, labels):
        self.session.run(self.train_step, feed_dict={self.input_vec: batch, self.target_q: labels})

    def predict(self, input):
        actions_q, best_action = self.session.run([self.q, self.selected_action], feed_dict={self.input_vec: input})
        return actions_q, best_action

    def save(self, episode):
        self.saver.save(self.session, self.save_name, global_step=episode)

    def load(self):
        self.saver.restore(self.session, self.load_name)
