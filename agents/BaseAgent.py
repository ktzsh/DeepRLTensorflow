import os
import yaml
import math
import pickle
import random
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import tensorflow as tf

from utils.History import History
from utils.ReplayMemory import ReplayMemory

class BaseAgent(object):
    def __init__(self, type, name, input_shape, nb_actions):

        self.ENV_TYPE = type
        with open('cfg/' + type + '.yml', 'rb') as stream:
            self.config = yaml.load(stream)

        self.ENV_NAME               = name
        self.DOUBLE_Q               = self.config['DOUBLE_Q']
        self.QUIET                  = self.config['QUIET']

        self.IMAGE_WIDTH            = self.config['IMAGE_WIDTH']
        self.IMAGE_HEIGHT           = self.config['IMAGE_HEIGHT']
        self.GRAYSCALE_IMG          = self.config['GRAYSCALE_IMG']
        self.NORMALIZE              = self.config['NORMALIZE_IMG']
        self.CROP                   = self.config['IMAGE_CROPING']

        self.TARGET_TICKS           = self.config['AGENT']['TARGET_TICKS']
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
        self.epsilon_step  = (self.INITIAL_EPSILON - self.FINAL_EPSILON) / (self.EXPLORATION_STEPS )

        # In case of exponential decay for epsilon this factor is required since value of epsilon has to
        # change from INITIAL to FINAL and not from 1.0 to 0.0 in update_explore_rate()
        self.correction    = 2 ** (self.INITIAL_EPSILON - self.FINAL_EPSILON) - 1.0

        self.total_reward  = 0.0
        self.total_q_max   = 0.0
        self.total_q_mean  = 0.0
        self.total_loss    = 0
        self.duration      = 0
        self.episode       = 0

        self.input_shape   = (self.STATE_LENGTH, ) + input_shape
        self.nb_actions    = nb_actions

        if type == "Atari": states_dtype = np.uint8
        else: states_dtype = np.float32
        self._history      = History(self.input_shape, states_dtype)

        if not os.path.exists(self.SAVE_NETWORK_PATH + self.ENV_NAME):
            os.makedirs(self.SAVE_NETWORK_PATH + self.ENV_NAME)
        if not os.path.exists(self.SAVE_SUMMARY_PATH + self.ENV_NAME):
            os.makedirs(self.SAVE_SUMMARY_PATH + self.ENV_NAME)
        if not os.path.exists(self.SAVE_TRAIN_STATE_PATH + self.ENV_NAME):
            os.makedirs(self.SAVE_TRAIN_STATE_PATH + self.ENV_NAME)

        restore_prefix = self.SAVE_TRAIN_STATE_PATH + self.ENV_NAME + '/'
        if os.path.exists(restore_prefix + 'snapshot.lock'):
            print "Restoring Training State Snapshot from", restore_prefix
            with open(restore_prefix + 'snapshot.lock', 'rb') as f:
                params = f.read().split('\n')
                self.episode = int(params[1])
                self.t       = int(params[2])
                self.epsilon = float(params[3])

            self._memory     = ReplayMemory(self.MEMORY_SIZE, self.input_shape[1:], self.STATE_LENGTH,
                                                                restore=restore_prefix, states_dtype=states_dtype)
            self.tb_counter  = len([log for log in os.listdir(os.path.expanduser(
                                            self.SAVE_SUMMARY_PATH + self.ENV_NAME)) if 'Experiment_' in log])
        else:
            self._memory     = ReplayMemory(self.MEMORY_SIZE, self.input_shape[1:], self.STATE_LENGTH,
                                                                states_dtype=states_dtype)
            self.tb_counter  = len([log for log in os.listdir(os.path.expanduser(
                                            self.SAVE_SUMMARY_PATH + self.ENV_NAME)) if 'Experiment_' in log]) + 1
            os.makedirs(self.SAVE_SUMMARY_PATH + self.ENV_NAME + '/Experiment_' + str(self.tb_counter))

        self.summary_writer = tf.summary.FileWriter(self.SAVE_SUMMARY_PATH + self.ENV_NAME + '/Experiment_'
                                                        + str(self.tb_counter), tf.get_default_graph())

        # Save snapshot of run configuration used
        with open(self.SAVE_SUMMARY_PATH + self.ENV_NAME + '/Experiment_'
                    + str(self.tb_counter) + '/config.run.yml', 'wb') as stream:
            yaml.dump(self.config, stream, default_flow_style=False)

    def build_network(self, input_shape):
        raise NotImplementedError

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
        self.q_value = self.q_network.predict([
                            history.reshape((1,) + history.shape),
                            np.ones(self.nb_actions).reshape(1, self.nb_actions)])[0]

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

        e_summary = tf.Summary(
            value=[tf.Summary.Value(
                tag=self.ENV_NAME + '/Epsilon/Timestep', simple_value=self.epsilon)])
        self.summary_writer.add_summary(e_summary, self.t)

        if done:
            # Write summary
            if self.t >= self.INITIAL_REPLAY_SIZE:
                self.write_summary()

            # Debug
            if self.t < self.INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif self.INITIAL_REPLAY_SIZE <= self.t < self.INITIAL_REPLAY_SIZE + self.EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'

            if not self.QUIET:
                print "-----EPISODE SUMMARY-----"
                print "EPISODE    :", self.episode + 1, \
                    "\nTIMESTEP   :", self.t, \
                    "\nDURATION   :", self.duration, \
                    "\nEPSILON    :", self.epsilon, \
                    "\nTOTALREWARD:", self.total_reward, \
                    "\nAVG_MAX_Q  :", self.total_q_max / float(self.duration), \
                    "\nAVG_MEAN_Q :", self.total_q_mean / float(self.duration), \
                    "\nAVG_LOSS   :", self.total_loss / float(self.duration), \
                    "\nMODE       :", mode
                print "-------------------------"

            self.total_reward = 0
            self.total_q_max  = 0
            self.total_q_mean = 0
            self.total_loss   = 0
            self.duration     = 0
            self.episode     += 1

            # Reset the short term memory
            self._history.reset()

        # Append to long term memory
        self._memory.append(old_state, action, reward, done)

        if self.SAVE_TRAIN_STATE and (self.t % self.SAVE_INTERVAL == 0 ):
            if os.path.exists(self.SAVE_TRAIN_STATE_PATH + self.ENV_NAME + '/snapshot.lock'):
                os.remove(self.SAVE_TRAIN_STATE_PATH + self.ENV_NAME + '/snapshot.lock')
            snapshot_params =  'Snapshot Parameters [Episode, Timestep, Actions Taken, Epsilon]\n' + \
                                        str(self.episode) + '\n' + \
                                        str(self.t) + '\n' + \
                                        str(self.epsilon) + '\n'
            self._memory.save(self.SAVE_TRAIN_STATE_PATH + self.ENV_NAME + '/')
            with open(self.SAVE_TRAIN_STATE_PATH + self.ENV_NAME + '/snapshot.lock', 'wb') as f:
                f.write(snapshot_params)

    def train(self):
        # Train network
        if (self.t % self.TRAIN_INTERVAL) == 0:
            self.train_network()

        # Update target network
        if self.t % self.TARGET_UPDATE_INTERVAL == 0:
            self.target_q_network.set_weights(self.q_network.get_weights())

        # Save network
        if self.t % self.SAVE_INTERVAL == 0:
            save_path = self.SAVE_NETWORK_PATH + self.ENV_NAME + '/chkpnt-' + str(self.t) + '.h5'
            self.q_network.save_weights(save_path)
            if not self.QUIET: print "Successfully saved:", save_path

    def write_summary(self):
        r_summary = tf.Summary(
            value=[tf.Summary.Value(
                tag=self.ENV_NAME + '/Total Reward/Episode', simple_value=self.total_reward)])
        self.summary_writer.add_summary(r_summary, self.episode + 1)

        q_summary = tf.Summary(
            value=[tf.Summary.Value(
                tag=self.ENV_NAME + '/Average Max Q/Episode', simple_value=(self.total_q_max / float(self.duration)))])
        self.summary_writer.add_summary(q_summary, self.episode + 1)

        l_summary = tf.Summary(
            value=[tf.Summary.Value(
                tag=self.ENV_NAME + '/Average Loss/Episode', simple_value=(self.total_loss / float(self.duration)))])
        self.summary_writer.add_summary(l_summary, self.episode + 1)

        d_summary = tf.Summary(
            value=[tf.Summary.Value(
                tag=self.ENV_NAME + '/Duration/Episode', simple_value=self.duration)])
        self.summary_writer.add_summary(d_summary, self.episode + 1)

        t_summary = tf.Summary(
            value=[tf.Summary.Value(
                tag=self.ENV_NAME + '/Timestep/Episode', simple_value=self.t)])
        self.summary_writer.add_summary(t_summary, self.episode + 1)

    def train_network(self):
        raise NotImplementedError

    def setup_summary(self):
        raise NotImplementedError

    def predict(self, state):
        self.t += 1
        self._history.append(state)

        if self.t >= self.STATE_LENGTH:
            history = self._history.value
            q_value = self.q_network.predict([
                                history.reshape((1,) + history.shape),
                                np.ones(self.nb_actions).reshape(1, self.nb_actions)])[0]
            action = np.argmax(q_value)
            return action
        else:
            return 0

    def load_network(self, weight_file=None):
        if weight_file is not None:
            self.q_network.load_weights(weight_file)
        else:
            chkpnts = [log for log in os.listdir(os.path.expanduser(
                                        self.SAVE_NETWORK_PATH + self.ENV_NAME)) if 'chkpnt-' in log]
            if len(chkpnts) != 0:
                chkpnts.sort()
                self.q_network.load_weights(self.SAVE_NETWORK_PATH + self.ENV_NAME + '/' + chkpnts[-1])
