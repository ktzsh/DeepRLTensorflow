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

        self.test      = self.config['TEST']
        self.display   = self.config['DISPLAY']

    def preprocess(self, state):
        return state

    def render(self):
        if self.display:
            self.env.render()

    def reset(self):
        state = self.preprocess(self.env.reset())
        self.render()
        return state

    def step(self, action):
        state, reward, terminal, _ = self.preprocess(self.env.step(action))

        print "Reward     :", reward
        print "Action     :", action
        print "Done       :", terminal

        self.render()
        return state, reward, (terminal*1.0)
