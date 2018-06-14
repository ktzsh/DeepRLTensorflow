# Human-Level Control through Deep Reinforcement Learning

Tensorflow implementation of [Human-Level Control through Deep Reinforcement Learning](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf).


This implementation contains:

1. DQN (Deep Q-Network) and DDQN (Double Deep Q-Network)
2. Experience Replay Memory
    - to reduce the correlations between consecutive updates
3. Network for Q-learning targets are fixed for intervals [OpenAI hack]
    - to reduce the correlations between target and predicted Q-values
4. Image Cropping and Explicit Frame Skiping as in Original paper

## Requirements

- Python 2.7 or Python 3.3+
- [gym](https://github.com/openai/gym)
- [OpenCV2](http://opencv.org/)
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [Keras](https://keras.io/)

## Usage

First, install prerequisites with:

    $ pip install gym[all]

To train a model for Breakout:
    - Edit the config.yml change 'NAME' to 'Breakout-v0'
    - Change Other parameters as desired
    $ python Agent.py


## Results



## TODOs
- [x] Implement DDQN
- [ ] Implement DRQN
- [ ] Add Detailed Results
- [ ] Add Test modes



## References

- [DQN Tensorflow Implementation](https://github.com/carpedm20/deep-rl-tensorflow)
- [DQN Tensorflow Implementation](https://github.com/devsisters/DQN-tensorflow)
- [Code for Human-level control through deep reinforcement learning](https://sites.google.com/a/deepmind.com/dqn/)


## License

MIT License.
