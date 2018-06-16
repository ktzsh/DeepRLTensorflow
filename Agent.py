import sys
import time
import yaml
import numpy as np

from Base import BaseAgent

with open("config.yml", 'r') as stream:
    config = yaml.load(stream)
    if config['ENVIRONMENT']['TYPE'] == "Atari":
        from EnvironmentAtari import Environment
    elif config['ENVIRONMENT']['TYPE'] == "Classic":
        from EnvironmentClassic import Environment

import tensorflow as tf
from keras.layers import Input, ConvLSTM2D, Dense, Flatten, Conv2D, Reshape
from keras.models import Model

class Agent(BaseAgent):
    def __init__(self, nb_actions):
        # Call Base Class init method
        super(Agent, self).__init__(nb_actions)

        # Action Value model (used by agent to interact with the environment)
        self.s, self.q_values, q_network = self.build_network(self.input_shape)
        q_network_weights                = q_network.trainable_weights
        q_network.summary()

        # Target model used to compute the target Q-values in training, updated
        # less frequently for increased stability.
        self.st, self.target_q_values, target_network = self.build_network(self.input_shape)
        target_network_weights                        = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [ target_network_weights[i].assign(q_network_weights[i])
                                            for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self.build_training_op(q_network_weights)

        self.sess  = tf.InteractiveSession()
        self.saver = tf.train.Saver(q_network_weights)

        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()

        self.summary_writer = tf.summary.FileWriter(self.SAVE_SUMMARY_PATH + self.ENV_NAME + '/Experiment_'
                                                        + str(self.tb_counter), self.sess.graph)

        self.sess.run(tf.initialize_all_variables())

        # Load network
        if self.LOAD_NETWORK:
            self.load_network()

        # Initialize target network
        self.sess.run(self.update_target_network)


    def build_network(self, input_shape):
        input_frames = Input(shape=input_shape, dtype='float32', name='input_frames')
        x = Reshape(input_shape[1:-1] + (input_shape[0] * input_shape[-1],))(input_frames)

        if self.ENV_TYPE == "Atari":
            x = Conv2D(32, (8, 8), strides=(4,4), activation='relu', name='conv1')(x)
            x = Conv2D(64, (4, 4), strides=(2,2), activation='relu', name='conv2')(x)
            x = Conv2D(64, (3, 3), strides=(1,1), activation='relu', name='conv2')(x)
            x = Flatten(name='flatten')(x)
            x = Dense(512, activation='relu', name='fc1')(x)
        elif self.ENV_TYPE == "Classic":
            x = Dense(32, activation='relu', name='fc1')(x)
            x = Dense(64, activation='relu', name='fc2')(x)
            x = Dense(64, activation='relu', name='fc3')(x)

        output = Dense(self.nb_actions, name='output')(x)
        model = Model(inputs=input_frames, outputs=output)

        s = tf.placeholder(tf.float32, (None,) + self.input_shape)
        q_values = model(s)

        return s, q_values, model

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.nb_actions, 1.0, 0.0)
        q_value   = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error          = tf.abs(y - q_value)
        clipped_error  = tf.where(error < 1.0, 0.5 * tf.square(error), error - 0.5)
        loss           = tf.reduce_mean(clipped_error)

        optimizer    = tf.train.RMSPropOptimizer(self.LEARNING_RATE, momentum=self.MOMENTUM, epsilon=self.MIN_GRAD)
        grads_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grads_update

    def train_network(self):
        ''' Extension to train() call - Batch generation and graph computations
        '''
        # Sample random minibatch of transition from replay memory
        state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = self._memory.minibatch(self.BATCH_SIZE)

        if self.DOUBLE_Q:
            q_values_batch = self.q_values.eval(feed_dict={self.st: next_state_batch})
            max_q_value_idx = np.argmax(q_values_batch, axis=1)
            target_q_values_at_idx_batch = self.target_q_values.eval(feed_dict={self.st: next_state_batch})[:, max_q_value_idx]
            y_batch = reward_batch + (1 - terminal_batch) * self.GAMMA * target_q_values_at_idx_batch
        else:
            target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: next_state_batch})
            y_batch = reward_batch + (1 - terminal_batch) * self.GAMMA * np.max(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.s: state_batch,
            self.a: action_batch,
            self.y: y_batch
        })
        self.total_loss += loss

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q    = tf.Variable(0.)
        episode_avg_mean_q   = tf.Variable(0.)
        episode_duration     = tf.Variable(0.)
        episode_avg_loss     = tf.Variable(0.)
        episode_timestep     = tf.Variable(0.)

        tf.summary.scalar(self.ENV_NAME + ':Total Reward/Episode', episode_total_reward)
        tf.summary.scalar(self.ENV_NAME + ':Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar(self.ENV_NAME + ':Average Mean Q/Episode', episode_avg_mean_q)
        tf.summary.scalar(self.ENV_NAME + ':Duration/Episode', episode_duration)
        tf.summary.scalar(self.ENV_NAME + ':Average Loss/Episode', episode_avg_loss)
        tf.summary.scalar(self.ENV_NAME + ':Timestep/Episode', episode_timestep)

        summary_vars         = [episode_total_reward, episode_avg_max_q, episode_avg_mean_q, episode_duration, episode_avg_loss, episode_timestep]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops           = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op           = tf.summary.merge_all()

        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(self.SAVE_NETWORK_PATH + self.ENV_NAME)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')


if __name__=="__main__":

    env   = Environment()
    agent = Agent(env.n)

    current_state = env.reset()
    if not agent.TEST:
        # Train
        while True:
            action                  = agent.act(current_state)
            new_state, reward, done = env.step(action)

            agent.observe(current_state, action, reward, done)
            agent.train()

            if done:
                print "Restarting the Game"
                new_state = env.reset()

            current_state = new_state
            print "--------------------\n"
    else:
        # Test
        while True:
            action = agent.test(current_state)
            new_state, _, _ = env.step(action)
            current_state = new_state
