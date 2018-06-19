import os
import gym
import cv2
import yaml
import random
import numpy as np
from gym import wrappers

class Environment(object):
    def __init__(self, type, name):
        with open('cfg/' + type + '.yml', 'rb') as stream:
            self.config = yaml.load(stream)

        monitor     = self.config['ENVIRONMENT']['ADD_MONITOR_WRAPPER']
        results_dir = self.config['ENVIRONMENT']['MONITOR_PATH'] + name
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        record_video_every = self.config['ENVIRONMENT']['RECORD_INTERVAL']

        self.env = gym.make(name)
        if monitor:
            self.env       = wrappers.Monitor(self.env,
                                    results_dir,
                                    video_callable=lambda count: count % record_video_every == 0,
                                    resume=True)

        self.name              = name
        self.type              = type
        self.action_space      = self.env.action_space
        self.observation_space = self.env.observation_space

        self.display   = self.config['DISPLAY']['RENDER']
        if type == "Atari":
            self.rescale   = self.config['DISPLAY']['RESCALE']
            self.scale_w   = self.config['DISPLAY']['SCALE_W']
            self.scale_h   = self.config['DISPLAY']['SCALE_H']

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
            if self.type == "Atari" and self.rescale:
                screen = self.env.render(mode='rgb_array')
                scaled = cv2.resize(screen, (0,0), fx=self.scale_w, fy=self.scale_h)
                cv2.imshow(self.name, scaled)
                cv2.waitKey(1)
            else:
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
