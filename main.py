import cv2
import sys

from env.Atari import AtariPlayer
from env.AtariWrappers import FrameStack, MapState, FireResetEnv

from agent.DQNAgent import DQNAgent as Agent

if __name__ == "__main__":
    ROM_FILE      = sys.argv[1]
    ACTION_REPEAT = 4
    viz           = 0.03
    train         = True
    IMAGE_SIZE    = (84, 84)

    env = AtariPlayer(ROM_FILE, frame_skip=ACTION_REPEAT, viz=viz,
                      live_lost_as_eoe=train, max_num_frames=60000)
    env = FireResetEnv(env)
    env = MapState(env, lambda im: cv2.resize(im, IMAGE_SIZE).reshape(IMAGE_SIZE + (1,)))

    agent = Agent("Atari", "Breakout", (84, 84, 1), 4)
    agent.learn(env)
