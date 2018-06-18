import gym
import cv2
import yaml
import random
import numpy as np

class Environment(object):
    def __init__(self, type, name):
        with open('cfg/' + type + '.yml', 'rb') as stream:
            self.config = yaml.load(stream)

        self.env               = gym.make(name)
        self.type              = type
        self.action_space      = self.env.action_space
        self.observation_space = self.env.observation_space

        self.display   = self.config['DISPLAY']
        self.im_width  = self.config['IMAGE_WIDTH']
        self.im_height = self.config['IMAGE_HEIGHT']
        self.grayscale = self.config['GRAYSCALE_IMG']
        self.normalize = self.config['NORMALIZE_IMG']
        self.crop      = self.config['IMAGE_CROPING']
        self.repeat    = self.config['REPEAT_ACTION']

        if type == "Atari":
            if self.grayscale:
                self.input_shape = (self.im_height, self.im_width, 1)
            else:
                self.input_shape = (self.im_height, self.im_width, 3)
        elif type == "Classic":
            self.input_shape = self.observation_space.shape

    def observation_shape(self):
        return self.input_shape

    def nb_actions(self):
        return self.env.action_space.n

    def preprocess(self, screen):
        if self.type == "Atari":
            channel = 3
            if self.crop:
                screen = screen[34:-16, :, :]
            if self.grayscale:
                channel = 1
                screen  = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (self.im_width, self.im_height))
            screen = np.asarray(screen, dtype='float32')
            if self.normalize:
                screen /= 255.0
            screen = screen.reshape((self.im_height, self.im_width, channel))
        # cv2.imshow('Debug', screen)
        # cv2.waitKey(0)
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

        if self.type == "Atari":
            start_lives = self.env.unwrapped.ale.lives()

        for i in range(random.choice(self.repeat)):
            screen, reward, terminal, _ = self.env.step(action)
            screen                      = self.preprocess(screen)
            cummulative_reward          = cummulative_reward + reward

            # if self.type == "Atari":
            #     if start_lives > self.env.unwrapped.ale.lives():
            #         cummulative_reward   = -10.0

            if terminal:
                break

        self.render()
        return screen, cummulative_reward, (terminal*1.0)
