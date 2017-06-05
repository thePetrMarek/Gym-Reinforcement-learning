import random

import gym
import tensorflow as tf
import numpy as np

episodes = 3000
epsilon = 0.0
epsilon_min = 0.0
gama = 0.9
epsilon_decay = (epsilon - epsilon_min) / episodes

# Neural network for Q approximation
input_vec = tf.placeholder(shape=[None, 4], dtype=tf.float32)
target_q = tf.placeholder(shape=[None, 2], dtype=tf.float32)

W1 = tf.Variable(tf.random_uniform([4, 128], 0, 0.01))
b1 = tf.Variable(tf.zeros([128]))
layer_one = tf.nn.relu(tf.matmul(input_vec, W1) + b1)

W2 = tf.Variable(tf.random_uniform([128, 128], 0, 0.01))
b2 = tf.Variable(tf.zeros([128]))
layer_two = tf.nn.relu(tf.matmul(layer_one, W2) + b2)

W3 = tf.Variable(tf.random_uniform([128, 2], 0, 0.01))
b3 = tf.Variable(tf.zeros([2]))
q = tf.matmul(layer_two, W3) + b3

selected_action = tf.argmax(q, 1)

loss = tf.reduce_mean(tf.square(q - target_q))
train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)

init = tf.global_variables_initializer()


def evaluate_actions(state, session):
    state = state.reshape(1, 4)
    actions_q, best_action = session.run([q, selected_action], feed_dict={input_vec: state})
    return actions_q, best_action


def epsilon_greedy_policy(state, session):
    actions_q, best_action = evaluate_actions(state, session)
    action = best_action[0]
    if random.random() < epsilon:
        action = random.randint(0, 1)
    return actions_q, action


def update():
    samples = get_from_memory()
    input = []
    predictions = []
    for sample in samples:
        state = sample[0]
        new_state = sample[3]
        action = sample[1]
        reward = sample[2]
        done = sample[4]

        state2d = state.reshape(1, 4)
        new_state2d = new_state.reshape(1, 4)
        actions_q, _ = evaluate_actions(state2d, session)
        next_actions_q, _ = evaluate_actions(new_state2d, session)
        next_state_value = np.max(next_actions_q)
        if not done:
            actions_q[0, action] = reward + gama * next_state_value
        else:
            actions_q[0, action] = reward
        target = actions_q
        input.append(state)
        predictions.append(target[0])
    session.run(train_step, feed_dict={input_vec: input, target_q: predictions})


moving_av = []

memory_size = 50
memory = []


def add_to_memory(sample):
    memory.append(sample)
    if len(memory) > memory_size:
        memory.pop(0)


def get_from_memory():
    n = min(50, len(memory))
    return random.sample(memory, n)


def moving_average(total_reward):
    moving_av.append(total_reward)
    if (len(moving_av) > 100):
        moving_av.pop(0)
    print(sum(moving_av) / len(moving_av))


# Start
env = gym.make('CartPole-v0')
saver = tf.train.Saver()
with tf.Session() as session:
    #session.run(init)
    saver.restore(session, '.\model\model-2900')

    for episode in range(episodes):
        total_reward = 0
        observation = env.reset()
        step = 0
        while True:
            #if episode %100 ==0:
            env.render()
            actions_q, action = epsilon_greedy_policy(observation, session)

            next_observation, reward, done, info = env.step(action)
            total_reward += reward

            add_to_memory((observation, action, reward, next_observation, done))

            observation = next_observation
            if done:
                #epsilon -= epsilon_decay
                print("Episode " + str(episode) + ": " + str(total_reward))
                moving_average(total_reward)
                break
            #update()
        #if episode % 100 == 0:
        #    saver.save(session, '.\model\model', global_step=episode)
    #saver.save(session, '.\model\model', global_step=episodes)


