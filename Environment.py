import gym
import cv2
import yaml
import numpy as np

class Environment(object):
    def __init__(self):
        with open("config.yml", 'r') as stream:
            self.config = yaml.load(stream)

        self.env       = gym.make(self.config['ENVIRONMENT']['NAME'])
        self.env.reset()

        self.test      = self.config['TEST']
        self.display   = self.config['DISPLAY']
        self.im_width  = self.config['IMAGE_WIDTH']
        self.im_height = self.config['IMAGE_HEIGHT']
        self.grayscale = self.config['GRAYSCALE_IMG']
        self.normalize = self.config['NORMALIZE_IMG']
        self.crop      = self.config['IMAGE_CROPING']
        self.repeat    = self.config['REPEAT_ACTION']

    def preprocess(self, screen):
        channel = 3
        if self.grayscale:
            channel = 1
            screen  = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        screen = cv2.resize(screen, (self.im_width, self.im_height))
        if self.crop:
            dim    = self.im_height - self.im_width
            screen = observation[dim:, :, :]

        screen = np.asarray(screen, dtype='float32')
        if self.normalize:
            screen /= 255.0
        screen = screen.reshape((self.im_height, self.im_width, channel))
        return screen

    def render(self):
        if self.display:
            self.env.render()

    def reset(self):
        if self.env.env.ale.lives() == 0:
            screen = self.preprocess(self.env.reset())
        else:
            screen, _, _, _ = self.env.step(0)
            screen = self.preprocess(screen)

        self.render()
        return screen

    def step(self, action):
        cummulative_reward = 0.0
        start_lives = self.env.env.ale.lives()

        for i in range(self.repeat):
            screen, reward, terminal, _ = self.env.step(action)
            screen                      = self.preprocess(screen)
            cummulative_reward          = cummulative_reward +  reward

            if (not self.test) and (start_lives > self.env.env.ale.lives()):
                terminal             = True
                cummulative_reward   = -1.0

            if terminal:
                break

        print "Reward     :", cummulative_reward
        print "Action     :", action
        print "Done       :", terminal

        self.render()
        return screen, cummulative_reward, (terminal*1.0)
