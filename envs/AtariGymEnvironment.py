import os
import gym
import cv2
import yaml
import random
import numpy as np
from gym import wrappers

from BaseGymEnvironment import BaseGymEnvironment

class AtariGymEnvironment(BaseGymEnvironment):
    def __init__(self, type, name):
        with open('cfg/' + type + '.yml', 'rb') as stream:
            self.config = yaml.load(stream)

        super(AtariGymEnvironment, self).__init__(type, name)

        self.lives         = self.env.unwrapped.ale.lives()
        self.start_lives   = self.lives

        self.random_game   = self.config['ENVIRONMENT']['RANDOM_GAME']
        self.random_starts = self.config['ENVIRONMENT']['RANDOM_STARTS']

        self.rescale   = self.config['DISPLAY']['RESCALE']
        self.scale_w   = self.config['DISPLAY']['SCALE_W']
        self.scale_h   = self.config['DISPLAY']['SCALE_H']

        self.im_width  = self.config['IMAGE_WIDTH']
        self.im_height = self.config['IMAGE_HEIGHT']
        self.grayscale = self.config['GRAYSCALE_IMG']
        self.normalize = self.config['NORMALIZE_IMG']
        self.crop      = self.config['IMAGE_CROPING']

        if self.grayscale:
            self.input_shape = (self.im_height, self.im_width, 1)
        else:
            self.input_shape = (self.im_height, self.im_width, 3)

    def observation_shape(self):
        return self.input_shape

    def nb_actions(self):
        return self.env.action_space.n

    def preprocess(self, screen):
        channel = 3
        if self.crop:
            screen = screen[34:-16, :, :]
        if self.grayscale:
            channel = 1
            screen  = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (self.im_width, self.im_height))
        screen = np.asarray(screen, dtype='uint8')
        screen = screen.reshape((self.im_height, self.im_width, channel))
        # cv2.imshow('Processed Screen', screen)
        # cv2.waitKey(1)
        return screen

    def render(self):
        if self.display:
            if self.rescale:
                screen = self.env.render(mode='rgb_array')
                scaled = cv2.resize(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB), (0,0), fx=self.scale_w, fy=self.scale_h)
                cv2.imshow(self.name, scaled)
                cv2.waitKey(1)
            else:
                self.env.render()

    def reset(self):
        if self.random_game:
            screen = self.env.reset()
            no_rnd = np.random.randint(1, self.random_starts)
            for i in range(no_rnd):
                screen, _, _, info = self.env.step(0)
            screen = self.preprocess(screen)
        else:
            screen = self.preprocess(self.env.reset())
        self.current_lives = self.lives
        self.render()
        return screen

    def step(self, action):
        terminal = False
        cummulative_reward = 0.0

        for i in range(random.choice(self.repeat)):
            screen, reward, terminal, info = self.env.step(action)
            screen                         = self.preprocess(screen)
            cummulative_reward             = cummulative_reward + reward

            lives = info['ale.lives']
            if lives < self.current_lives:
                # cummulative_reward = -1.0
                self.current_lives = lives

            if terminal:
                # cummulative_reward = -1.0
                break

        self.render()
        return screen, cummulative_reward, (terminal*1.0)
