import os
import sys
import time
import yaml
import numpy as np
from collections import deque

from BaseAgent import BaseAgent

import keras.backend as K
from keras.layers import Input, ConvLSTM2D, Dense, Flatten, Conv2D, Reshape, Lambda, Multiply
from keras.models import Model, clone_model
from keras.initializers import TruncatedNormal, Constant
from keras.optimizers import RMSprop, Adam


class DQNAgent(BaseAgent):
    def __init__(self, type, name, input_shape, action_space):
        # Call Base Class init method
        super(DQNAgent, self).__init__(type, name, input_shape, action_space)

        # Action Value model (used by agent to interact with the environment)
        self.q_network = self.build_network()

        # Target model used to compute the target Q-values in training, updated
        # less frequently for increased stability.
        self.target_q_network = clone_model(self.q_network)

        # Load network
        if self.LOAD_NETWORK:
            self.load_network()

        # Initialize target network
        self.target_q_network.set_weights(self.q_network.get_weights())


    def build_network(self):
        def huber_loss(y, q_value):
            error = K.abs(y - q_value)
            quadratic_part = K.clip(error, 0.0, 1.0)
            linear_part = error - quadratic_part
            loss = K.mean(0.5 * K.square(quadratic_part) + linear_part - 0.5)
            return loss

        input_shape   = self.input_shape
        input_frames  = Input(shape=input_shape, dtype='float32')
        input_actions = Input((self.nb_actions,))

        x = Reshape(input_shape[1:-1] + (input_shape[0] * input_shape[-1],))(input_frames)
        if self.ENV_TYPE == "Atari":
            x = Lambda(lambda x: x / 255.0)(x)
            x = Conv2D(32, (8, 8), strides=(4,4),
                        activation='relu')(x)
            x = Conv2D(64, (4, 4), strides=(2,2),
                        activation='relu')(x)
            x = Conv2D(64, (3, 3), strides=(1,1),
                        activation='relu')(x)
            x = Flatten()(x)
            x = Dense(512, activation='relu')(x)
        elif self.ENV_TYPE == "Classic":
            x = Dense(64, activation='tanh')(x)
            x = Dense(64, activation='tanh')(x)

        output = Dense(self.nb_actions, activation='linear')(x)
        filtered_output = Multiply()([output, input_actions])


        model     = Model(inputs=[input_frames, input_actions], outputs=filtered_output)
        if self.USE_ADAPTIVE:
            optimizer = Adam(lr=self.LEARNING_RATE, clipnorm=self.GRADIENT_CLIP_NORM)
        else:
            optimizer = RMSprop(lr=self.LEARNING_RATE, rho=self.RHO, epsilon=self.EPSILON, clipnorm=self.GRADIENT_CLIP_NORM)

        model.compile(optimizer, loss=huber_loss)
        model.summary()
        return model

    def train_network(self):
        ''' Extension to train() call - Batch generation and graph computations
        '''
        # Sample random minibatch of transition from replay memory
        state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = self._memory.minibatch(self.BATCH_SIZE)

        action_masks = np.ones((self.BATCH_SIZE, self.nb_actions))
        if self.DOUBLE_Q:
            q_values_batch  = self.q_network.predict([next_state_batch, action_masks])
            max_q_value_idx = np.argmax(q_values_batch, axis=1)
            target_q_values_at_idx_batch = self.target_q_network.predict([next_state_batch, action_masks])[range(self.BATCH_SIZE), max_q_value_idx]
            y_batch = reward_batch + (1 - terminal_batch) * self.GAMMA * target_q_values_at_idx_batch
        else:
            target_q_values_batch = self.target_q_network.predict([next_state_batch, action_masks])
            y_batch = reward_batch + (1 - terminal_batch) * self.GAMMA * np.max(target_q_values_batch, axis=1)

        action_one_hot_batch = np.eye(self.nb_actions)[action_batch.reshape(-1)]
        target_one_hot_batch = action_one_hot_batch * y_batch[:, None]

        loss = self.q_network.train_on_batch([state_batch, action_one_hot_batch], target_one_hot_batch)
        self.total_loss += loss

    def play(self, env):
        current_state = env.reset()
        for i in range(self.STATE_LENGTH-1):
            _ = self.predict(current_state)

        while True:
            action = self.predict(current_state)
            new_state, reward, done = env.step(action)
            current_state = new_state
            if done:
                break

    def learn(self, env):
        scores = deque(maxlen=100)

        # Agent Observes the environment - no ops steps
        print "Warming Up..."
        current_state = env.reset()
        for i in range(self.INITIAL_REPLAY_SIZE):

            action = self.act(current_state)
            new_state, reward, done = env.step(action)
            self.observe(current_state, action, reward, done)

            current_state = new_state
            if done:
                current_state = env.reset()

        # Actual Training Begins
        print "Begin Training..."
        for i in range(self.MAX_EPISODES):
            t    = 0
            done = False
            while not done:
                action                  = self.act(current_state)
                new_state, reward, done = env.step(action)
                self.observe(current_state, action, reward, done)
                self.train()

                if not self.QUIET:
                    print "Reward     :", reward
                    print "Action     :", action
                    print "Done       :", terminal

                current_state = new_state
                t += 1
            current_state = env.reset()

            scores.append(t)
            mean_score = np.mean(scores)

            if i % 100 == 0 and i != 0:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks'.format(str(i).zfill(6), mean_score))
                if (self.TARGET_TICKS is not None) and mean_score >= self.TARGET_TICKS:
                    print "Mean target over last 100 episodes reached."
                    break

        self.summary_writer.close()
        env.close()
