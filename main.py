import os
import argparse

from tensorpack.utils import logger
from tensorpack import OfflinePredictor, PredictConfig, SimpleTrainer, get_model_loader, launch_train_with_config

from common.common import eval_model_multithread, play_n_episodes
from environment.atari import AtariPlayer
from agent.DQN import get_config, get_player, Model
from cfg import AtariConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument(
        '--task',
        help='task to perform',
        choices=['play', 'eval', 'train'],
        default='train')
    parser.add_argument('--rom', help='atari rom', required=True)
    parser.add_argument(
        '--algo',
        help='algorithm',
        choices=['DQN', 'Double', 'Dueling'],
        default='Double')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    ROM_FILE = args.rom
    METHOD = args.algo

    # set num_actions
    NUM_ACTIONS = AtariPlayer(ROM_FILE).action_space.n
    logger.info("ROM: {}, Num Actions: {}".format(ROM_FILE, NUM_ACTIONS))

    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(
            PredictConfig(
                model=Model(NUM_ACTIONS, METHOD),
                session_init=get_model_loader(args.load),
                input_names=['state'],
                output_names=['Qvalue']))
        if args.task == 'play':
            play_n_episodes(get_player(
                ROM_FILE, viz=AtariConfig.VIZ_SCALE), pred, 100)
        elif args.task == 'eval':
            eval_model_multithread(
                pred, AtariConfig.EVAL_EPISODE, get_player, ROM_FILE)
    else:
        logger.set_logger_dir(
            os.path.join('logs', 'DQN-{}'.format(
                os.path.basename(ROM_FILE).split('.')[0])))
        config = get_config(ROM_FILE, NUM_ACTIONS, METHOD)
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SimpleTrainer())
