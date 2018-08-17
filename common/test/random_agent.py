import sys
import numpy as np

from env.Atari import AtariPlayer

if __name__ == '__main__':
    env = AtariPlayer(sys.argv[1], viz=0.03, viz_scale=3.0)
    num = env.action_space.n
    while True:
        action = np.random.randint(num)
        state, reward, isOver, info = env.step(action)
        if isOver:
            print "****", info, "****"
            env.reset()
        print "Reward:", reward, "Action:", action, "IsOver:", isOver
