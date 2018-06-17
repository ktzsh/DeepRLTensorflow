import gym
import cv2
import yaml
import random
import numpy as np

class Environment(object):
    def __init__(self):
        with open("config.yml", 'r') as stream:
            self.config = yaml.load(stream)

        self.env = gym.make(self.config['ENVIRONMENT']['NAME'])
        self.n   = self.env.action_space.n

        self.display   = self.config['DISPLAY']


    def render(self):
        if self.display:
            self.env.render()

    def reset(self):
        state = self.env.reset()
        self.render()
        return state

    def step(self, action):
        state, reward, terminal, _ = self.env.step(action)

        # if terminal:
        #     reward = -100.0

        self.render()
        return state, reward, (terminal*1.0)
