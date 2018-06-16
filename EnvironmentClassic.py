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
        self.im_width  = self.config['IMAGE_WIDTH']
        self.im_height = self.config['IMAGE_HEIGHT']
        self.grayscale = self.config['GRAYSCALE_IMG']
        self.normalize = self.config['NORMALIZE_IMG']
        self.crop      = self.config['IMAGE_CROPING']
        self.repeat    = self.config['REPEAT_ACTION']

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
