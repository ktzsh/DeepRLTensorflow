import os
import yaml
import math
import random
import numpy as np

from collections import deque

from agent.DQNModel import DQNModel as Model

from common.History import History
from common.ReplayMemory import ReplayMemory


class DQNAgent(object):
    def __init__(self, type, name, input_shape, nb_actions):

        self.ENV_TYPE = type
        with open('cfg/' + type + '.yml', 'rb') as stream:
            self.config = yaml.load(stream)

        self.ENV_NAME               = name
        self.DOUBLE_Q               = self.config['DOUBLE_Q']
        self.DUELING                = self.config['DUELING']
        self.QUIET                  = self.config['QUIET']

        self.IMAGE_WIDTH            = self.config['IMAGE_WIDTH']
        self.IMAGE_HEIGHT           = self.config['IMAGE_HEIGHT']
        self.GRAYSCALE_IMG          = self.config['GRAYSCALE_IMG']
        self.NORMALIZE              = self.config['NORMALIZE_IMG']
        self.CROP                   = self.config['IMAGE_CROPING']

        self.TARGET_SCORE           = self.config['AGENT']['TARGET_SCORE']
        self.MAX_EPISODES           = self.config['AGENT']['MAX_EPISODES']
        self.STATE_LENGTH           = self.config['AGENT']['STATE_LENGTH']
        self.GAMMA                  = self.config['AGENT']['GAMMA']
        self.EXPLORATION_STEPS      = self.config['AGENT']['EXPLORATION_STEPS']
        self.INITIAL_EPSILON        = self.config['AGENT']['INITIAL_EPSILON']
        self.FINAL_EPSILON          = self.config['AGENT']['FINAL_EPSILON']
        self.EPSILON_EXP_DECAY      = self.config['AGENT']['EPSILON_EXP_DECAY']
        self.INITIAL_REPLAY_SIZE    = self.config['AGENT']['INITIAL_REPLAY_SIZE']
        self.MEMORY_SIZE            = self.config['AGENT']['MEMORY_SIZE']
        self.BATCH_SIZE             = self.config['AGENT']['BATCH_SIZE']
        self.TARGET_UPDATE_INTERVAL = self.config['AGENT']['TARGET_UPDATE_INTERVAL']
        self.TRAIN_INTERVAL         = self.config['AGENT']['TRAIN_INTERVAL']
        self.SAVE_INTERVAL          = self.config['AGENT']['SAVE_INTERVAL']
        self.LOAD_NETWORK           = self.config['AGENT']['LOAD_NETWORK']
        self.SAVE_NETWORK_PATH      = self.config['AGENT']['SAVE_NETWORK_PATH']
        self.SAVE_SUMMARY_PATH      = self.config['AGENT']['SAVE_SUMMARY_PATH']
        self.SAVE_TRAIN_STATE       = self.config['AGENT']['SAVE_TRAIN_STATE']
        self.SAVE_TRAIN_STATE_PATH  = self.config['AGENT']['SAVE_TRAIN_STATE_PATH']

        self.LEARNING_RATE          = self.config['AGENT']['OPTIMIZER']['LEARNING_RATE']
        self.USE_ADAPTIVE           = self.config['AGENT']['OPTIMIZER']['USE_ADAPTIVE']
        self.GRADIENT_CLIP_NORM     = self.config['AGENT']['OPTIMIZER']['GRADIENT_CLIP']
        self.RHO                    = self.config['AGENT']['OPTIMIZER']['RHO']
        self.EPSILON                = self.config['AGENT']['OPTIMIZER']['EPSILON']
        self.DECAY_RATE             = self.config['AGENT']['OPTIMIZER']['DECAY_RATE']

        self.t             = 0
        self.epsilon       = self.INITIAL_EPSILON
        self.epsilon_step  = (self.INITIAL_EPSILON - self.FINAL_EPSILON) / (self.EXPLORATION_STEPS)

        # In case of exponential decay for epsilon this factor is required since value of epsilon has to
        # change from INITIAL to FINAL and not from 1.0 to 0.0 in update_explore_rate()
        self.correction    = 2 ** (self.INITIAL_EPSILON - self.FINAL_EPSILON) - 1.0

        self.total_reward  = 0.0
        self.total_q_max   = 0.0
        self.total_q_mean  = 0.0
        self.total_loss    = 0.0
        self.duration      = 0
        self.episode       = 0

        self.input_shape   = (self.STATE_LENGTH, ) + input_shape
        self.nb_actions    = nb_actions

        self._history      = History(self.input_shape)
        self._memory       = ReplayMemory(self.MEMORY_SIZE, self.input_shape[1:], self.STATE_LENGTH)

        network_path = self.SAVE_NETWORK_PATH + self.ENV_NAME + "/"
        summary_path = self.SAVE_SUMMARY_PATH + self.ENV_NAME + "/"
        if not os.path.exists(network_path):
            os.makedirs(network_path)
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)

        # Create DQN Model
        self.model = Model(self.input_shape, self.nb_actions, self.ENV_NAME, lr=self.LEARNING_RATE,
                        epsilon=self.EPSILON, dueling=self.DUELING, double_q=self.DOUBLE_Q,
                        use_adaptive=self.USE_ADAPTIVE, load_network=self.LOAD_NETWORK,
                        network_path=network_path, summary_path=summary_path)

    def update_explore_rate(self):
        if self.EPSILON_EXP_DECAY:
            self.epsilon = max(self.FINAL_EPSILON, min(self.INITIAL_EPSILON, 1.0 - math.log(1.0 + self.correction * ((self.t - self.INITIAL_REPLAY_SIZE) / float(self.EXPLORATION_STEPS)), 2.0)))
        else:
            self.epsilon = max(self.FINAL_EPSILON, self.epsilon - self.epsilon_step)

    def act(self, state):
        """ This allows the agent to select the next action to perform in regard of the current state of the environment.
        It follows the terminology used in the Nature paper.
        """
        # Append the state to the short term memory (ie. History)
        self._history.append(state)

        history = self._history.value
        self.q_value = self.model.predict_q_value(history)

        if np.random.rand() <= self.epsilon or self.t < self.INITIAL_REPLAY_SIZE:
            # Choose an action randomly
            action = random.randrange(self.nb_actions)
        else:
            # Use the network to output the best action
            action  = np.argmax(self.q_value)

        # Anneal epsilon linearly over time
        if self.epsilon > self.FINAL_EPSILON and self.t >= self.INITIAL_REPLAY_SIZE:
            self.update_explore_rate()

        self.t += 1
        return action

    def observe(self, old_state, action, reward, done):
        """ This allows the agent to observe the output of doing the action it selected through act() on the old_state
        """
        self.total_reward += reward
        self.total_q_max += np.max(self.q_value)
        self.duration += 1

        if done:
            # Write summary
            if self.t >= self.INITIAL_REPLAY_SIZE:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                            self.total_q_mean / float(self.duration), self.duration,
                            self.total_loss / (float(self.duration)), self.epsilon, self.t]
                self.model.write_summary(stats, self.episode)
            # Debug
            if self.t < self.INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif self.INITIAL_REPLAY_SIZE <= self.t < self.INITIAL_REPLAY_SIZE + self.EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'

            if not self.QUIET:
                print "EPISODE:", self.episode + 1, \
                      "\tTIMESTEP:", self.t, \
                      "\tDURATION:", self.duration, \
                      "\tEPSILON:", self.epsilon, \
                      "\tTOTALREWARD:", self.total_reward, \
                      "\tMODE:", mode

            self.total_reward = 0.0
            self.total_q_max  = 0.0
            self.total_q_mean = 0.0
            self.total_loss   = 0.0
            self.duration     = 0
            self.episode     += 1

            # Reset the short term memory
            self._history.reset()

        # Append to long term memory
        self._memory.append(old_state, action, reward, done)

    def train(self):
        # Train network
        if (self.t % self.TRAIN_INTERVAL) == 0:
            state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = self._memory.minibatch(self.BATCH_SIZE)
            loss = self.model.train_network(state_batch, action_batch, next_state_batch, reward_batch, terminal_batch)
            self.total_loss += loss

        # Update target network
        if self.t % self.TARGET_UPDATE_INTERVAL == 0:
            self.model.update_target_network_op()

        # Save network
        if self.t % self.SAVE_INTERVAL == 0:
            self.model.save_network(self.t)

    def learn(self, env):
        scores = deque(maxlen=100)

        # Agent Observes the environment - no ops steps
        print "Warming Up..."
        current_state = env.reset()
        for i in range(self.INITIAL_REPLAY_SIZE):

            action = self.act(current_state)
            new_state, reward, done, info = env.step(action)
            self.observe(current_state, action, reward, done)

            current_state = new_state
            if done:
                current_state = env.reset()

        # Actual Training Begins
        print "Begin Training..."
        for i in range(self.MAX_EPISODES):
            total_reward = 0
            done = False
            while not done:
                action = self.act(current_state)
                new_state, reward, done, _ = env.step(action)
                self.observe(current_state, action, reward, done)
                self.train()

                current_state = new_state
                total_reward += reward
            current_state = env.reset()

            scores.append(total_reward)
            mean_score = np.mean(scores)

            if i % 100 == 0 and i != 0:
                print('[Episode {}] - Mean Reward over last 100 episodes was {}'.format(str(i).zfill(6), mean_score))
                if (self.TARGET_SCORE is not None) and mean_score >= self.TARGET_SCORE:
                    print "Mean target over last 100 episodes reached."
                    break
