import os
import yaml
import pickle
import random
import numpy as np

from utils.History import History
from utils.ReplayMemory import ReplayMemory

class BaseAgent(object):
    def __init__(self, nb_actions):

        with open("config.yml", 'r') as stream:
            self.config = yaml.load(stream)

        self.AGENT_TYPE             = self.config['AGENT_TYPE']
        self.DOUBLE_Q               = self.config['DOUBLE_Q']

        self.TEST                   = self.config['TEST']

        self.ENV_NAME               = self.config['ENVIRONMENT']['NAME']

        self.STATE_LENGTH           = self.config['AGENT']['STATE_LENGTH']
        self.GAMMA                  = self.config['AGENT']['GAMMA']
        self.EXPLORATION_STEPS      = self.config['AGENT']['EXPLORATION_STEPS']
        self.INITIAL_EPSILON        = self.config['AGENT']['INITIAL_EPSILON']
        self.FINAL_EPSILON          = self.config['AGENT']['FINAL_EPSILON']
        self.INITIAL_REPLAY_SIZE    = self.config['AGENT']['INITIAL_REPLAY_SIZE']
        self.MEMORY_SIZE            = self.config['AGENT']['MEMORY_SIZE']
        self.BATCH_SIZE             = self.config['AGENT']['BATCH_SIZE']
        self.TARGET_UPDATE_INTERVAL = self.config['AGENT']['TARGET_UPDATE_INTERVAL']
        self.TRAIN_INTERVAL         = self.config['AGENT']['TRAIN_INTERVAL']
        self.LEARNING_RATE          = self.config['AGENT']['LEARNING_RATE']
        self.MOMENTUM               = self.config['AGENT']['MOMENTUM']
        self.MIN_GRAD               = self.config['AGENT']['MIN_GRAD']
        self.SAVE_INTERVAL          = self.config['AGENT']['SAVE_INTERVAL']
        self.LOAD_NETWORK           = self.config['AGENT']['LOAD_NETWORK']
        self.SAVE_NETWORK_PATH      = self.config['AGENT']['SAVE_NETWORK_PATH']
        self.SAVE_SUMMARY_PATH      = self.config['AGENT']['SAVE_SUMMARY_PATH']
        self.SAVE_MEMORY            = self.config['AGENT']['SAVE_MEMORY']

        self.IMAGE_WIDTH   = self.config['IMAGE_WIDTH']
        self.IMAGE_HEIGHT  = self.config['IMAGE_HEIGHT']
        self.GRAYSCALE_IMG = self.config['GRAYSCALE_IMG']
        self.NORMALIZE     = self.config['NORMALIZE_IMG']
        self.CROP          = self.config['IMAGE_CROPING']

        self.t             = 0
        self.epsilon       = self.INITIAL_EPSILON
        self.epsilon_step  = (self.INITIAL_EPSILON - self.FINAL_EPSILON) / (self.EXPLORATION_STEPS * self.STATE_LENGTH)

        self.total_reward  = 0.0
        self.total_q_max   = 0.0
        self.total_loss    = 0
        self.duration      = 0
        self.episode       = 0

        self.nb_actions    = nb_actions
        if self.CROP:
            self.IMAGE_HEIGHT = self.IMAGE_WIDTH

        if self.GRAYSCALE_IMG:
            self.input_shape = (self.STATE_LENGTH, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1)
        else:
            self.input_shape = (self.STATE_LENGTH, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3)

        self._history           = History(self.input_shape)
        self._num_actions_taken = 0

        self.pickle_path        = None
        if os.path.exists('pickle.txt'):
            path = None
            with open('pickle.txt', 'rb') as f:
                path   = f.readline().rstrip('\n')
                params = path[:-7].split('_')

                self.episode            = int(params[1])
                self.t                  = int(params[2])
                self._num_actions_taken = int(params[3])
                self.epsilon            = float(params[4])
            with open(path, 'rb') as f:
                print "Restoring Replay Memory from", path
                self._memory     = pickle.load(f)
                self.pickle_path = path
                self.tb_counter  = len([log for log in os.listdir(os.path.expanduser(
                                            self.SAVE_SUMMARY_PATH + self.ENV_NAME)) if 'Experiment_' in log])
        else:
            self._memory     = ReplayMemory(self.MEMORY_SIZE, self.input_shape[1:], self.STATE_LENGTH)
            self.tb_counter  = len([log for log in os.listdir(os.path.expanduser(
                                            self.SAVE_SUMMARY_PATH + self.ENV_NAME)) if 'Experiment_' in log]) + 1

        if not os.path.exists(self.SAVE_NETWORK_PATH + self.ENV_NAME):
            os.makedirs(self.SAVE_NETWORK_PATH + self.ENV_NAME)
        if not os.path.exists(self.SAVE_SUMMARY_PATH + self.ENV_NAME):
            os.makedirs(self.SAVE_SUMMARY_PATH + self.ENV_NAME)

    def build_network(self, input_shape):
        raise NotImplementedError

    def build_training_op(self, q_network_weights):
        raise NotImplementedError

    def act(self, state):
        """ This allows the agent to select the next action to perform in regard of the current state of the environment.
        It follows the terminology used in the Nature paper.
        """
        # Append the state to the short term memory (ie. History)
        self._history.append(state)

        if self.epsilon >= random.random() or self.t < self.INITIAL_REPLAY_SIZE:
            # Choose an action randomly
            action = random.randrange(self.nb_actions)
        else:
            # Use the network to output the best action
            env_with_history = self._history.value
            action = np.argmax(self.q_values.eval(feed_dict={self.s: env_with_history.reshape((1,) + env_with_history.shape)}))

        # Anneal epsilon linearly over time
        if self.epsilon > self.FINAL_EPSILON and self.t >= self.INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        # Keep track of interval action counter
        self._num_actions_taken += 1
        return action

    def observe(self, old_state, action, reward, done):
        """ This allows the agent to observe the output of doing the action it selected through act() on the old_state
        """
        # If done, reset short term memory (ie. History)
        self.total_reward += reward
        env_with_history = self._history.value
        self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: env_with_history.reshape((1,) + env_with_history.shape)}))
        self.duration += 1

        if done:
            # Write summary
            if self.t >= self.INITIAL_REPLAY_SIZE:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                        self.duration, self.total_loss / (float(self.duration)), self.t]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            if self.t < self.INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif self.INITIAL_REPLAY_SIZE <= self.t < self.INITIAL_REPLAY_SIZE + self.EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print "-----EPISODE SUMMARY-----"
            print "EPISODE    :", self.episode + 1, \
                "\nTIMESTEP   :", self.t, \
                "\nDURATION   :", self.duration, \
                "\nEPSILON    :", self.epsilon, \
                "\nTOTALREWARD:", self.total_reward, \
                "\nAVG_MAX_Q  :", self.total_q_max / float(self.duration), \
                "\nAVG_LOSS   :", self.total_loss / float(self.duration), \
                "\nMODE       :", mode

            print "-------------------------"

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

            # Reset the short term memory
            self._history.reset()

        # Append to long term memory

        self._memory.append(old_state, action, reward, done)
        if done and self.SAVE_MEMORY:
            old_pickle_path = self.pickle_path
            self.pickle_path = 'memory_' + str(self.episode) + '_' + str(self.t) + '_' \
                            + str(self._num_actions_taken) + '_' + str(self.epsilon) + \
                            '.pickle'
            self._memory.save(self.pickle_path)

            if old_pickle_path is not None:
                os.remove(old_pickle_path)

    def train(self):
        """ This allows the agent to train itself to better understand the environment dynamics.
        The agent will compute the expected reward for the state(t+1)
        and update the expected reward at step t according to this.

        The target expectation is computed through the Target Network, which is a more stable version
        of the Action Value Network for increasing training stability.

        The Target Network is a frozen copy of the Action Value Network updated as regular intervals.
        """
        agent_step = self._num_actions_taken
        # if agent_step >= self.TRAIN_AFTER:
        if (agent_step % self.TRAIN_INTERVAL) == 0:
            # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
            # reward = np.clip(reward, -1, 1)
            print "Episode    :", self.episode, \
                "\nTimestep   :", self.t, \
                "\nAgent Step :", agent_step

            if self.t >= self.INITIAL_REPLAY_SIZE:
                # Train network
                self.train_network()

                # Update target network
                if self.t % self.TARGET_UPDATE_INTERVAL == 0:
                    self.sess.run(self.update_target_network)

                # Save network
                if self.t % self.SAVE_INTERVAL == 0:
                    save_path = self.saver.save(self.sess, self.SAVE_NETWORK_PATH + self.ENV_NAME + '/chkpnt', global_step=self.t)
                    print "Successfully saved:", save_path

            self.t += 1

    def train_network(self):
        raise NotImplementedError

    def setup_summary(self):
        raise NotImplementedError

    def load_network(self):
        raise NotImplementedError

    def test(self, state):
        self.t += 1
        self._history.append(state)

        if self.t >= self.STATE_LENGTH:
            env_with_history = self._history.value
            action = np.argmax(self.q_values.eval(feed_dict={self.s: env_with_history.reshape((1,) + env_with_history.shape)}))
            return action
        else:
            return 0
