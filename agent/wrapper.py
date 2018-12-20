import os
import sys
import gym

from common.loader import get_config
from agent.translator import export_config

from baselines import deepq
from baselines import bench
from baselines.common.atari_wrappers import make_atari

import models.custom_model

class BaseAgent(object):
    def __init__(self, config, env=None):
        self.env = env
        self.config = config

    def learn(self):
        raise NotImplementedError

    def play(self):
        raise NotImplementedError


class OpenAIAgent(BaseAgent):
    def __init__(self, config, env=None):
        from baselines import logger
        self.logger = logger
        super(OpenAIAgent, self).__init__(config, env)

    def get_player(self, train=False):
        if self.env:
            return env

        if self.config['ENV_TYPE'] == 'Classic':
            env = gym.make(self.config['ENV_NAME'])
        elif self.config['ENV_TYPE'] == 'Atari':
            if train:
                env = make_atari(self.config['ENV_NAME'])
                env = bench.Monitor(env, self.logger.get_dir())
                env = deepq.wrap_atari_dqn(env)
            else:
                env = gym.make(self.config['ENV_NAME'])
                env = deepq.wrap_atari_dqn(env)
        else:
            raise Exception('Environment Type %s - Not Supported' % self.config['ENV_TYPE'])
        return env

    def learn(self):
        def callback(lcl, _glb):
            is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
            return is_solved

        self.logger.configure()

        config = self.config
        env = self.get_player(train=True)
        model = deepq.learn(
            env,
            config['MODEL']['TYPE'],
            **config['MODEL']['ARGS'],
            **config['LOAD_PATH'],
            dueling=config['DUELING'],
            lr=config['LEARNING_RATE'],
            total_timesteps=config['TOTAL_TIMESTEPS'],
            buffer_size=config['BUFFER_SIZE'],
            exploration_fraction=config['EXPLORATION_FRACTION'],
            exploration_final_eps=config['EXPLORATION_FINAL_EPS'],
            train_freq=config['TRAIN_FREQ'],
            learning_starts=config['NO_OP_STEPS'],
            target_network_update_freq=config['TARGET_UPDATE_FREQ'],
            gamma=config['GAMMA'],
            seed=config['SEED'],
            batch_size=config['BATCH_SIZE'],
            print_freq=config['PRINT_FREQ'],
            checkpoint_freq=config['CHECKPOINT_FREQ'],
            checkpoint_path=config['CHECKPOINT_PATH_PREFIX'],
            prioritized_replay=config['PRIORITIZED_REPLAY'],
            prioritized_replay_alpha=config['PRIORITIZED_REPLAY_ALPHA'],
            prioritized_replay_beta0=config['PRIORITIZED_REPLAY_BETA'],
            prioritized_replay_beta_iters=config['PRIORITIZED_REPLAY_BETA_ITERS'],
            prioritized_replay_eps=config['PRIORITIZED_REPLAY_EPS'],
            param_noise=config['PARAM_NOISE'],
            callback=callback
        )

        model.save(config['CHECKPOINT_PATH_PREFIX'] + config['ENV_NAME'] + '.pkl')
        env.close()

    def play(self):
        config = self.config

        env = self.get_player()
        model = deepq.learn(
            env,
            config['MODEL']['TYPE'],
            **config['MODEL']['ARGS'],
            **config['LOAD_PATH'],
            dueling=config['DUELING'],
            total_timesteps=0
        )

        while True:
            obs, done = env.reset(), False
            episode_rew = 0
            while not done:
                env.render()
                obs, rew, done, _ = env.step(model(obs[None])[0])
                episode_rew += rew
            print("Episode reward", episode_rew)


def create_agent(args, custom_env=None):
    config = get_config(args.config)
    config = export_config(config, args.agent)

    agent_class = getattr(sys.modules[__name__], args.agent + 'Agent')
    return agent_class(config, custom_env)
