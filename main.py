import argparse

from agents.DQNAgent import DQNAgent
from envs.AtariGymEnvironment import AtariGymEnvironment
from envs.BaseGymEnvironment import BaseGymEnvironment

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["Atari", "Classic"],
                                  help="Select the Type of Game from OpenAI gym",
                                  required=True)
    parser.add_argument("--name", help="Select the Name of Game eg. Breakout-v0",
                                  required=True)
    parser.add_argument("--mode", choices=["train", "test"], help="Choose to Train or Test", default="train",
                                  required=False)
    args = parser.parse_args()

    if args.type == "Classic":
        environment  = BaseGymEnvironment(args.type, args.name)
    elif args.type == "Atari":
        environment  = AtariGymEnvironment(args.type, args.name)

    input_shape  = environment.observation_shape()
    nb_actions   = environment.nb_actions()

    agent = DQNAgent(args.type, args.name, input_shape, nb_actions)
    if args.mode == "train":
        agent.learn(environment)
    else:
        agent.play(environment)
