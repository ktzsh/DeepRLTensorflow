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
- [OpenCV2](http://opencv.org/)
- [TensorFlow](https://github.com/tensorflow/tensorflow) (Only for tensorboard)
- [Keras](https://keras.io/)

## Usage

1. First, install prerequisites with:

        $ pip install gym[all]

2. To train a model for Breakout(or Any Other Atari Games):
    - Edit the config.yml as required and run:

            $ python main.py --type Atari --name BreakoutDeterministic-v4

3. To train a model for CartPole(or Any Other Classic Control Games):
    - Edit the config.yml as required and run:

            $ python main.py --type Classic --name CartPole-v0

4. To test a model run:

        $ python main.py --type <TYPE> --name <NAME> --mode test


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
