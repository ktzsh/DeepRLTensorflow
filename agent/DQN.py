#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu
# Source: https://github.com/tensorpack/tensorpack/blob/master/examples/DeepQNetwork

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf

from tensorpack import argscope
from tensorpack import PReLU, Conv2D, LinearWrap, FullyConnected
from tensorpack import QueueInput, ModelSaver, PeriodicTrigger, RunOp, ScheduledHyperParamSetter, ObjAttrParam, PeriodicTrigger, HumanHyperParamSetter, TrainConfig

from agent.DQNModel import Model as DQNModel
from common.common import Evaluator
from environment.atari_wrapper import FrameStack, MapState, FireResetEnv
from common.expreplay import ExpReplay
from environment.atari import AtariPlayer
from cfg import AtariConfig


def get_player(ROM_FILE, viz=False, train=False):
    env = AtariPlayer(ROM_FILE, frame_skip=AtariConfig.ACTION_REPEAT, viz=viz,
                      live_lost_as_eoe=train, max_num_frames=60000)
    env = FireResetEnv(env)
    env = MapState(env, lambda im: cv2.resize(
        im, AtariConfig.IMAGE_SIZE)[:, :, np.newaxis])
    if not train:
        # in training, history is taken care of in expreplay buffer
        env = FrameStack(env, AtariConfig.FRAME_HISTORY)
    return env


class Model(DQNModel):
    def __init__(self, NUM_ACTIONS, METHOD):
        super(Model, self).__init__(AtariConfig.IMAGE_SIZE, 1,
                                    AtariConfig.FRAME_HISTORY, METHOD, NUM_ACTIONS, AtariConfig.GAMMA)

    def _get_DQN_prediction(self, image):
        image = image / 255.0
        with argscope(Conv2D, activation=lambda x: PReLU('prelu', x), use_bias=True):
            l = (LinearWrap(image)
                 # Nature architecture
                 .Conv2D('conv0', 32, 8, strides=4)
                 .Conv2D('conv1', 64, 4, strides=2)
                 .Conv2D('conv2', 64, 3)

                 # architecture used for the figure in the README, slower but takes fewer iterations to converge
                 # .Conv2D('conv0', out_channel=32, kernel_shape=5)
                 # .MaxPooling('pool0', 2)
                 # .Conv2D('conv1', out_channel=32, kernel_shape=5)
                 # .MaxPooling('pool1', 2)
                 # .Conv2D('conv2', out_channel=64, kernel_shape=4)
                 # .MaxPooling('pool2', 2)
                 # .Conv2D('conv3', out_channel=64, kernel_shape=3)

                 .FullyConnected('fc0', 512)
                 .tf.nn.leaky_relu(alpha=0.01)())
        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, self.num_actions)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1)
            As = FullyConnected('fctA', l, self.num_actions)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.identity(Q, name='Qvalue')


def get_config(ROM_FILE, NUM_ACTIONS, METHOD):
    expreplay = ExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        player=get_player(ROM_FILE, train=True),
        state_shape=AtariConfig.IMAGE_SIZE + (1,),
        batch_size=AtariConfig.BATCH_SIZE,
        memory_size=AtariConfig.MEMORY_SIZE,
        init_memory_size=AtariConfig.INIT_MEMORY_SIZE,
        init_exploration=1.0,
        update_frequency=AtariConfig.UPDATE_FREQ,
        history_len=AtariConfig.FRAME_HISTORY
    )

    return TrainConfig(
        data=QueueInput(expreplay),
        model=Model(NUM_ACTIONS, METHOD),
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(
                RunOp(DQNModel.update_target_param, verbose=True),
                every_k_steps=AtariConfig.TARGET_NETWORK_UPDATE_FREQ),    # update target network every 10k steps
            expreplay,
            ScheduledHyperParamSetter('learning_rate',
                                      [(60, 4e-4), (100, 2e-4), (500, 5e-5)]),
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                # 1->0.1 in the first million steps
                [(0, 1), (10, 0.1), (320, 0.01)],
                interp='linear'),
            PeriodicTrigger(Evaluator(
                AtariConfig.EVAL_EPISODE, ['state'], ['Qvalue'], get_player, ROM_FILE),
                every_k_epochs=10),
            HumanHyperParamSetter('learning_rate'),
        ],
        steps_per_epoch=AtariConfig.STEPS_PER_EPOCH,
        max_epoch=AtariConfig.MAX_EPOCHS,
    )
