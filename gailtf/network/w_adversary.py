from gym.spaces import Discrete

from gailtf.baselines.common.mpi_running_mean_std import RunningMeanStd
from gailtf.baselines.common import tf_util as U
from gailtf.common.tf_util import *
import numpy as np


class TransitionClassifier(object):
    def __init__(self,
                 env,
                 hidden_size,
                 discriminatorStepSize=3e-4,
                 entcoeff=0.001,
                 scope="adversary"):

        global old_gen_loss, old_exp_loss
        print("Init Wasserstein discriminator")
        self.scope = scope
        self.observation_shape = env.observation_space.shape
        self.actions_shape = env.action_space.shape
        self.input_shape = tuple([o + a for o, a in zip(self.observation_shape, self.actions_shape)])

        self.num_actions = env.action_space.n if isinstance(env.action_space, Discrete) else env.action_space.shape[0]

        self.hidden_size = hidden_size
        self.discriminatorStepSize = discriminatorStepSize

        # PLACEHOLDERS
        self.generator_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape, name="observations_ph")
        self.generator_acs_ph = tf.placeholder(tf.float32, (None,) + self.actions_shape, name="actions_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape, name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(tf.float32, (None,) + self.actions_shape, name="expert_actions_ph")
        # Build graph
        gen_logits = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
        exp_logits = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)
        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(gen_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(exp_logits) > 0.5))

        # regression losses to control progress:
        old_gen_loss = regression_loss(gen_logits)
        old_exp_loss = regression_loss(exp_logits)

        # NR1. Use Wasserstein loss
        discriminator_loss = tf.contrib.gan.losses.wargs.wasserstein_discriminator_loss(exp_logits, gen_logits)
        # --- not sure about this loss function, but it doesn't take part in calculations:
        generator_loss = - tf.reduce_mean(gen_logits)

        # Build entropy loss
        logits = tf.concat([gen_logits, exp_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy
        # Loss + Accuracy terms
        self.losses = [generator_loss,
                       discriminator_loss,
                       old_gen_loss,
                       old_exp_loss,
                       entropy,
                       entropy_loss,
                       generator_acc,
                       expert_acc,
                       discriminator_loss + entropy_loss]
        self.loss_name = ["gen_loss",
                          "disc_loss",
                          "old_gen_loss",
                          "old_exp_loss",
                          "entropy",
                          "entropy_loss",
                          "generator_acc",
                          "expert_acc",
                          "total_loss"]

        self.total_loss = discriminator_loss + entropy_loss
        # Build Reward for policy
        self.reward_op = -tf.log(1 - tf.nn.sigmoid(gen_logits) + 1e-8)

        # NR2. Use RMSPropOptimizer
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=discriminatorStepSize).minimize(self.total_loss,
                                                                                                 var_list=self.get_trainable_variables())

        # NR3. Clip weights in range [-.01, .01]
        clip_ops = []
        for var in self.get_trainable_variables():
            clip_bounds = [-.01, .01]
            clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
        self.clip_disc_weights = tf.group(*clip_ops)

        self.dict = [self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph]

        # ================================ FUNCTIONS =====================================
        self.disc_train_op = U.function(self.dict, self.optimizer)

        self.losses = U.function(self.dict, self.losses)

        self.get_expert_logits = U.function([self.expert_obs_ph, self.expert_acs_ph], exp_logits)

        self.get_logits = U.function(self.dict, [exp_logits] + [gen_logits])

        self.clip = U.function(self.dict, self.clip_disc_weights)

    def optimize(self, gen_obs, gen_acs, exp_obs, exp_acs, clip=True):
        losses = self.losses(gen_obs, gen_acs, exp_obs, exp_acs)
        self.disc_train_op(gen_obs, gen_acs, exp_obs, exp_acs)
        if clip:
            print("Clipping weights...")
            self.clip(gen_obs, gen_acs, exp_obs, exp_acs)
        return losses

    def build_graph(self, obs_ph, acs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)

            obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
            _input = tf.concat([obs, acs_ph], axis=1)  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, acs):
        sess = U.get_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)
        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: acs}
        reward = sess.run(self.reward_op, feed_dict)
        return reward

    def sync(self):
        return True
