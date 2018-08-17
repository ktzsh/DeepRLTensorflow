# Human-Level Control through Deep Reinforcement Learning

Tensorflow implementation of [Human-Level Control through Deep Reinforcement Learning](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf).


This implementation contains:

1. DQN (Deep Q-Network) and DDQN (Double Deep Q-Network)
2. Experience Replay Memory
    - to reduce the correlations between consecutive updates
3. Network for Q-learning targets are fixed for intervals [OpenAI hack]
    - to reduce the correlations between target and predicted Q-values
4. Image Cropping and Explicit Frame Skiping as in Original paper
5. Support for Both Atari and Classic Control Environemnts from OpenAI gym

## Requirements

- Python 2.7 or Python 3.3+
- Yaml
- [gym](https://github.com/openai/gym)
- [ALEInterface](https://github.com/mgbellemare/Arcade-Learning-Environment)
- [OpenCV2](http://opencv.org/)
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [Tensorboard]

## Usage

1. First, install prerequisites with:

        $ pip install gym[all]

2. Setup ALE Arcade-Learning-Environment
    - Install ALE from https://github.com/mgbellemare/Arcade-Learning-Environment:
    - Download ROM bin files from https://github.com/openai/atari-py/tree/master/atari_py/atari_roms:

            $ wget https://github.com/openai/atari-py/raw/master/atari_py/atari_roms/breakout.bin

3. To train a model for Breakout(or Any Other Atari Games):
    - Edit the cfg/Atari.yml as required and run:

            $ python main.py --rom <ROM bin file>

4. To test a model run:

        $ python main.py --rom <ROM bin file> --mode test


## Results



## TODOs
- [x] Implement DQN
- [x] Implement DDQN
- [ ] Adaptive Exploration Rates
- [ ] Implement DRQN
- [ ] Prioritized Experience Replay
- [ ] Add Detailed Results
- [ ] Dueling Network Architectures



## References

- [DQN Tensorflow Implementation](https://github.com/carpedm20/deep-rl-tensorflow)
- [DQN Tensorflow Implementation](https://github.com/devsisters/DQN-tensorflow)
- [Code for Human-level control through deep reinforcement learning](https://sites.google.com/a/deepmind.com/dqn/)


## License

MIT License.
