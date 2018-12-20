import sys
import argparse

from agent.wrapper import create_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', help='Agent implementation to be used',
                            choices=['OpenAI'], default='OpenAI')
    parser.add_argument('--config', help='Path to config file', default='cfg/atari.yaml')
    parser.add_argument('--task', help='Task to perform',
                        choices=['play', 'train'], default='train')
    args = parser.parse_args()

    agent = create_agent(args)
    if args.task == 'train':
        agent.learn()
    else:
        agent.play()
