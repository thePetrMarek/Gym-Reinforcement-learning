import random

import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

episodes = 500
epsilon = 0.2
epsilon_min = 0.0
gama = 0.9
epsilon_decay = (epsilon - epsilon_min) / episodes


# Neural network for Q approximation
input_vec = tf.placeholder(shape=[1, 16], dtype=tf.float32)
target_q = tf.placeholder(shape=[1, 2], dtype=tf.float32)

W1 = tf.Variable(tf.random_normal([16, 128], 0, 0.01))
b1 = tf.Variable(tf.zeros([128]))
layer_one = tf.nn.relu(tf.matmul(input_vec, W1) + b1)

W2 = tf.Variable(tf.random_normal([128, 128], 0, 0.01))
b2 = tf.Variable(tf.zeros([128]))
layer_two = tf.nn.relu(tf.matmul(layer_one, W2) + b2)

W2 = tf.Variable(tf.random_normal([128, 2], 0, 0.01))
b2 = tf.Variable(tf.zeros([2]))
q = tf.matmul(layer_two, W2) + b2

selected_action = tf.argmax(q, 1)

loss = tf.reduce_mean(tf.square(q - target_q))
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

init = tf.global_variables_initializer()


def evaluate_actions(state, session):
    state = state.reshape(1, 16)
    actions_q, best_action = session.run([q, selected_action], feed_dict={input_vec: state})
    return actions_q, best_action


def epsilon_greedy_policy(state, session):
    actions_q, best_action = evaluate_actions(state, session)
    action = best_action[0]
    if random.random() < epsilon:
        action = random.randint(0, 1)
    return actions_q, action


def update(state, reward, action, new_state, session):
    state = state.reshape(1, 16)
    new_state = new_state.reshape(1, 16)
    actions_q, _ = evaluate_actions(state, session)
    next_actions_q, _ = evaluate_actions(new_state, session)
    next_state_value = np.max(next_actions_q)
    actions_q[0, action] = reward + gama * next_state_value
    target = actions_q
    session.run(train_step, feed_dict={input_vec: state, target_q: target})


def initialize_state(observation):
    observation = np.append(observation, [0, 0, 0, 0])
    observation = np.append(observation, [0, 0, 0, 0])
    observation = np.append(observation, [0, 0, 0, 0])
    return observation


def add_observation(state, observation):
    new_state = state.copy()
    new_state = new_state[:-4]
    new_state = np.append([observation], new_state)
    return new_state


# Start
env = gym.make('CartPole-v0')
with tf.Session() as session:
    session.run(init)
    for episode in range(episodes):
        total_reward = 0
        observation = env.reset()
        state = initialize_state(observation)
        step = 0
        while True:
            env.render()
            actions_q, action = epsilon_greedy_policy(state, session)

            if step < 4:
                action = random.randint(0, 1)
                step += 1

            next_observation, reward, done, info = env.step(action)
            total_reward += reward
            next_state = add_observation(state, next_observation)

            if step >= 4:
                update(state, reward, action, next_state, session)

            state = next_state
            if done:
                epsilon -= epsilon_decay
                print("Episode " + str(episode) + ": " + str(total_reward))
                plt.scatter(episode, total_reward)
                plt.pause(0.01)
                break
