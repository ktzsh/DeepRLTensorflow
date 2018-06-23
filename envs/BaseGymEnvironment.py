import os
import gym
import cv2
import yaml
import random
import numpy as np
from gym import wrappers

class BaseGymEnvironment(object):
    def __init__(self, type, name):
        with open('cfg/' + type + '.yml', 'rb') as stream:
            self.config = yaml.load(stream)

        self.name    = name
        self.type    = type

        self.env     = gym.make(name)
        self.display = self.config['DISPLAY']['RENDER']
        monitor      = self.config['ENVIRONMENT']['ADD_MONITOR_WRAPPER']
        results_dir  = self.config['ENVIRONMENT']['MONITOR_PATH'] + name
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        if monitor:
            self.env = wrappers.Monitor(self.env,
                            results_dir,
                            force=True)

        self.repeat = self.config['REPEAT_ACTION']

        self.action_space      = self.env.action_space
        self.observation_space = self.env.observation_space
        self.input_shape       = self.observation_space.shape


    def observation_shape(self):
        return self.input_shape

    def nb_actions(self):
        return self.env.action_space.n

    def preprocess(self, screen):
        return screen

    def render(self):
        if self.display:
            self.env.render()

    def reset(self):
        screen = self.preprocess(self.env.reset())
        self.render()
        return screen

    def step(self, action):
        terminal = False
        cummulative_reward = 0.0

        for i in range(random.choice(self.repeat)):
            screen, reward, terminal, info = self.env.step(action)
            screen                         = self.preprocess(screen)
            cummulative_reward             = cummulative_reward + reward

            if terminal:
                # cummulative_reward = -1.0
                break

        self.render()
        return screen, cummulative_reward, (terminal*1.0)

    def close(self):
        self.env.monitor.close()
