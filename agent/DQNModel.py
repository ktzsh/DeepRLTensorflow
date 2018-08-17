import os
import numpy as np

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


# from keras.layers import Input, ConvLSTM2D, Dense, Flatten, Conv2D, Reshape, Lambda, Add, PReLU, LeakyReLU
# from keras.models import Model


class DQNModel(object):
    def __init__(self, input_shape, nb_actions, env_name, lr=1e-3, epsilon=1e-5, gamma=0.95,
                    dueling=False, double_q=False, use_adaptive=False, load_network=False,
                    network_path="chkpnts/", summary_path="logs/"):
        self.nb_actions = nb_actions
        self.lr, self.epsilon, self.gamma = lr, epsilon, gamma
        self.dueling, self.double_q, self.use_adaptive = dueling, double_q, use_adaptive
        tf.reset_default_graph()

        # Action Value model (used by agent to interact with the environment)
        self.s, self.q_values, q_network = self.build_network(input_shape)
        q_network_weights                = q_network.trainable_weights
        q_network.summary()

        # Target model used to compute the target Q-values in training, updated
        # less frequently for increased stability.
        self.st, self.target_q_values, target_network = self.build_network(input_shape)
        target_network_weights                        = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i])
                                            for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self.build_training_op(q_network_weights)

        self.sess  = tf.Session()
        self.saver = tf.train.Saver(q_network_weights)

        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary(env_name)
        self.tb_counter = len([log for log in os.listdir(os.path.expanduser(
                                        summary_path)) if 'Experiment_' in log]) + 1
        self.summary_writer = tf.summary.FileWriter(summary_path + 'Experiment_' +
                                        str(self.tb_counter), tf.get_default_graph())

        self.sess.run(tf.global_variables_initializer())

        self.network_path = network_path
        # Load network
        if load_network:
            self.load_network(network_path)

        # Initialize target network
        self.update_target_network_op()

    def build_network(self, input_shape):
        input_frames = tf.keras.layers.Input(shape=input_shape, dtype='float32')

        x = tf.keras.layers.Reshape(input_shape[1:-1] + (input_shape[0] * input_shape[-1],))(input_frames)
        x = tf.keras.layers.Lambda(lambda x: x / 255.0)(x)
        x = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), use_bias=True)(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), use_bias=True)(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), use_bias=True)(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(512, activation='linear')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

        if self.dueling:
            x_V  = tf.keras.layers.Dense(1, activation='linear')(x)
            x_As = tf.keras.layers.Dense(self.nb_actions, activation='linear')(x)
            # output = Add()([x_As, x_V - K.mean(x_As, axis=1, keep_dims=True)])
            output = tf.add(x_As, x_V - tf.reduce_mean(x_As, 1, keep_dims=True))
        else:
            output = tf.keras.layers.Dense(self.nb_actions, activation='linear')(x)

        model = tf.keras.Model(inputs=input_frames, outputs=output)

        s = tf.placeholder(tf.float32, (None,) + input_shape)
        q_values = model(s)
        return s, q_values, model

    def optimizer(self):
        if self.use_adaptive:
            return tf.train.AdamOptimizer(self.lr)
        else:
            return tf.train.RMSPropOptimizer(self.lr, epsilon=self.epsilon)

    def update_target_network_op(self):
        self.sess.run(self.update_target_network)

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.nb_actions, 1.0, 0.0)
        q_value   = tf.reduce_sum(self.q_values * a_one_hot, reduction_indices=1)

        # Use huber loss
        loss = tf.losses.huber_loss(
            y, q_value, reduction=tf.losses.Reduction.MEAN)

        optimizer     = self.optimizer()
        grads_update  = optimizer.minimize(loss, var_list=q_network_weights)
        return a, y, loss, grads_update

    def train_network(self, state_batch, action_batch, next_state_batch, reward_batch, terminal_batch):
        # Sample random minibatch of transition from replay memory
        # state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = self._memory.minibatch(self.BATCH_SIZE)
        reward_batch = np.clip(reward_batch, -1, 1)

        if self.double_q:
            q_values_batch        = self.sess.run(self.q_values, feed_dict={self.s: next_state_batch})
            greedy_choice         = np.argmax(q_values_batch, axis=1)
            predict_onehot        = np.eye(self.nb_actions)[greedy_choice]
            target_q_values_batch = self.sess.run(self.target_q_values, feed_dict={self.st: next_state_batch})
            best_q_values         = np.sum(target_q_values_batch * predict_onehot, axis=1)
        else:
            target_q_values_batch = self.sess.run(self.target_q_values, feed_dict={self.st: next_state_batch})
            best_q_values         = np.max(target_q_values_batch, axis=1)

        y_batch = reward_batch + (1 - terminal_batch) * self.gamma * best_q_values
        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.s: state_batch,
            self.a: action_batch,
            self.y: y_batch
        })
        # self.total_loss += loss
        return loss

    def predict_q_value(self, state):
        return self.sess.run(self.q_values, feed_dict={self.s: state.reshape((1,) + state.shape)})

    def setup_summary(self, env_name):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q    = tf.Variable(0.)
        episode_avg_mean_q   = tf.Variable(0.)
        episode_duration     = tf.Variable(0.)
        episode_avg_loss     = tf.Variable(0.)
        episode_epsilon      = tf.Variable(0.)
        episode_timestep     = tf.Variable(0.)

        tf.summary.scalar(env_name + '/Total Reward/Episode', episode_total_reward)
        tf.summary.scalar(env_name + '/Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar(env_name + '/Average Mean Q/Episode', episode_avg_mean_q)
        tf.summary.scalar(env_name + '/Duration/Episode', episode_duration)
        tf.summary.scalar(env_name + '/Average Loss/Episode', episode_avg_loss)
        tf.summary.scalar(env_name + '/EpsilonAtEndOf/Episode', episode_epsilon)
        tf.summary.scalar(env_name + '/Timestep/Episode', episode_timestep)

        summary_vars         = [episode_total_reward,
                                episode_avg_max_q,
                                episode_avg_mean_q,
                                episode_duration,
                                episode_avg_loss,
                                episode_epsilon,
                                episode_timestep]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops           = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op           = tf.summary.merge_all()

        return summary_placeholders, update_ops, summary_op

    def write_summary(self, stats, episode):
        # assert len(stats) == len(summary_placeholders)
        for i in range(len(stats)):
            self.sess.run(self.update_ops[i], feed_dict={
                self.summary_placeholders[i]: float(stats[i])
            })
        summary_str = self.sess.run(self.summary_op)
        self.summary_writer.add_summary(summary_str, episode + 1)

    def save_network(self, global_step):
        save_path = self.saver.save(self.sess, self.network_path + "model.ckpt", global_step=global_step)
        print "Successfully saved:", save_path

    def load_network(self, path):
        checkpoint = tf.train.get_checkpoint_state(path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')
