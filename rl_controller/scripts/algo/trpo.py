import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.regularizers import l2
from keras.models import model_from_json, load_model
import json
from algo.brain import NeuralNetwork


class TRPO(NeuralNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._define_params()

        ''' build graph '''
        self.model_path = kwargs['model_path']
        self.training_index = kwargs['training_index']
        self._init_placeholders()
        if self.training_index == None:
            self._build_models()
        else:
            self.load_network(self.training_index)

        ''' loss function and optimize operation'''
        self.neg_policy_loss = self._policy_loss_function()
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.policy_learning_rate)
        self.optimize_policy_op = self.optimizer.minimize(
            self.neg_policy_loss, var_list=self._mean_model_params.append(self.sigma))

        ''' sampling actions operation'''
        if self.policy_action_type == 'Continous':
            self.sampled_action = tf.squeeze(
                self.mu + self.sigma * tf.random.normal(shape=tf.shape(self.mu)))
        elif self.policy_action_type == 'Discrete':
            #TODO
            pass

    ''' Hyperparameters from Appendix A in https://arxiv.org/abs/1707.06347,
    with some changes while experimenting '''
    def _define_params(self):
        self.policy_learning_rate = 4 * 1e-06
        self.value_learning_rate = 1.5 * 1e-04
        self.n_policy_epochs = 20
        self.n_value_epochs = 15
        self.value_batch_size = 512
        self.kl_target = 0.003
        self.beta = 1
        self.beta_max = 20
        self.beta_min = 1/20
        self.ksi = 10
        self.reward_discount = 0.995
        self.gae_discount = 0.975
        self.traj_batch_size = 8
        self.activation = 'tanh'
        '''
        Policy Type: 'MLP' or 'RBF'
        MLP for fully-connected policy
        RBF for RBF policy. For details, see https://arxiv.org/pdf/1703.02660.pdf
        '''
        self.policy_type = 'MLP'  # {'RBF','MLP'} #TODO: Add RBF policy
        self.policy_action_type = 'Continous'  # {'discrete','Continous'}

    def _init_placeholders(self):
        self.r = tf.compat.v1.placeholder(
            shape=[None, 1], dtype='float32', name='rewards')
        self.actions_ph = tf.compat.v1.placeholder(
            'float32', [None, self.env.get_num_action()], name="actions")
        self.advantages_ph = tf.placeholder(
            'float32', [None, 1], name="GAE_advantages")
        self.prev_mu_ph = tf.placeholder(
            'float32', [None, self.env.get_num_action()], name="prev_iteration_mu")
        self.prev_sigma_ph = tf.placeholder(
            'float32', [None, self.env.get_num_action()], name="prev_iteration_sigma")
        self.beta_ph = tf.placeholder(
            shape=[], dtype='float32', name='beta_2nd_loss')
        self.ksi_ph = tf.placeholder(
            shape=[], dtype='float32', name='eta_3rd_loss')

    ''' note that adding action sigma network had bad performance. thus omitted. '''
    def _build_models(self):
        print("BUILDING MODEL")
        if self.policy_action_type == 'Continous':
            ''' action mean network '''
            mu_model_input = Input(tensor=self.input_ph)
            mu_model = Dense(units=256, activation=self.activation,
                            kernel_initializer=RandomNormal(0, 0.1))(mu_model_input)
            mu_model = Dense(units=128, activation=self.activation,
                            kernel_initializer=RandomNormal(0, 0.1))(mu_model)
            mu_model = Dense(units=128, activation=self.activation,
                            kernel_initializer=RandomNormal(0, 0.1))(mu_model)
            mean = Dense(units=self.env_action_number, activation=None,
                        kernel_initializer=RandomNormal())(mu_model)
        elif self.policy_action_type == 'Discrete':
            #TODO: Build policy model
            pass

        ''' state value network '''
        value_model_input = Input(batch_shape=(
            None, self.env_state_shape[0]))
        value_model = Dense(units=128,
                            activation=self.activation,
                            kernel_regularizer=l2(0.01))(value_model_input)
        value_model = Dense(units=128,
                            activation=self.activation,
                            kernel_regularizer=l2(0.01))(value_model)
        val = Dense(units=1, activation=None, kernel_initializer=RandomNormal(
            0, 0.1), kernel_regularizer=l2(0.01))(value_model)

        ''' policy models '''
        self.policy_mu_model = Model(inputs=[mu_model_input], outputs=[mean])

        ''' value model, updated using keras routine'''
        self.value_model = Model(inputs=[value_model_input], outputs=[val])
        adam_optimizer = Adam(lr=self.value_learning_rate,
                              beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.value_model.compile(
            loss='mean_squared_error', optimizer=adam_optimizer)

        ''' model outputs '''
        self.mu = self.policy_mu_model(self.input_ph)
        self.value = self.value_model(self.input_ph)
        self.sigma = tf.compat.v1.get_variable('sigma', (1, self.env_action_number),
                                               tf.float32,
                                               tf.constant_initializer(0.6))

        ''' trainable weights; defined here since no direct use required'''
        self._mean_model_params = self.policy_mu_model.trainable_weights
        self._value_model_params = self.value_model.trainable_weights

        print('\033[92m'+"POLICY MODEL STRUCTURE"+'\033[0m')
        self.policy_mu_model.summary()
        print('\033[92m'+"VALUE MODEL STRUCTURE"+'\033[0m')
        self.value_model.summary()

        return self

    def _policy_loss_function(self):
        normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        prev_normal_dist = tf.contrib.distributions.Normal(
            self.prev_mu_ph, self.prev_sigma_ph)

        self.logp = (normal_dist.log_prob(self.actions_ph))
        self.prev_logp = (prev_normal_dist.log_prob(self.actions_ph))
        self.kl_divergence = tf.contrib.distributions.kl_divergence(
            normal_dist, prev_normal_dist)
        self.entropy = tf.reduce_mean(
            tf.reduce_sum(normal_dist.entropy(), axis=1))

        ''' adaptive KL penalty coefficient 
        see p.12 Algorithm 3 in https://arxiv.org/pdf/1707.02286.pdf and https://arxiv.org/abs/1707.06347'''
        negloss = -tf.reduce_mean(self.advantages_ph *
                                  tf.exp(self.logp - self.prev_logp))
        negloss += tf.reduce_mean(self.beta_ph * self.kl_divergence)
        negloss += tf.reduce_mean(self.ksi_ph * tf.square(
            tf.maximum(0.0, self.kl_divergence - 2 * self.kl_target)))
        return negloss

    def sample_action(self, session, inpt, exploring):
        if exploring:
            return self.env.action_space.sample()
        else:
            return session.run(self.sampled_action, feed_dict={self.input_ph: np.reshape(inpt, (-1, self.env_state_shape[0]))})

    def _update_policy(self, session, t, auditor):
        states = t['states']
        actions = t['actions']
        advantages = t['advantages']

        feed_dict = {self.input_ph: states,
                     self.actions_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.ksi_ph: self.ksi}
        prev_mu, prev_sigma = session.run([self.mu, self.sigma], feed_dict)
        feed_dict[self.prev_mu_ph] = np.reshape(
            prev_mu, (-1, self.env.get_num_action()))
        feed_dict[self.prev_sigma_ph] = np.reshape(
            prev_sigma, (-1, self.env.get_num_action()))

        neg_policy_loss, kl_divergence, entropy = 0.0, 0.0, 0.0
        to_fetch = [self.neg_policy_loss, self.kl_divergence, self.entropy]
        for _ in range(self.n_policy_epochs):
            session.run(self.optimize_policy_op,
                        feed_dict=feed_dict)  # update policy
            neg_policy_loss, kl_divergence, entropy = session.run(
                to_fetch, feed_dict=feed_dict)
            kl_divergence = np.mean(kl_divergence)
            if kl_divergence > 4 * self.kl_target:
                break

        auditor.update({'policy_loss': float("%.5f" % -neg_policy_loss),
                        'kl_divergence': float("%.4f" % kl_divergence),
                        'beta': self.beta,
                        'entropy': float("%.5f" % entropy)})

        ''' p.4 in https://arxiv.org/pdf/1707.06347.pdf '''
        if kl_divergence < self.kl_target / 1.5:
            self.beta /= 2
        elif kl_divergence > self.kl_target * 1.5:
            self.beta *= 2
        self.beta = np.clip(self.beta, self.beta_min, self.beta_max)

    def _update_value(self, t, auditor):
        states = t['states']
        target_values = t['disc_rewards']
        self.value_model.fit(x=states, y=target_values,
                             epochs=self.n_value_epochs,
                             batch_size=self.value_batch_size,
                             verbose=0)

        value_loss = self.value_model.evaluate(
            x=states, y=target_values, verbose=0)
        auditor.update({'value_loss': float("%.7f" % value_loss)})
        return self

    def save_network(self, index):
        policy_json = self.policy_mu_model.to_json()
        value_json = self.value_model.to_json()

        with open(self.model_path+"policy_mu_model_"+str(index)+".json","w") as policy_file:
            policy_file.write(policy_json)
        self.policy_mu_model.save_weights(self.model_path+"policy_mu_model_"+str(index)+".h5")

        with open(self.model_path+"value_model_"+str(index)+".json","w") as value_file:
            value_file.write(value_json)
        self.value_model.save_weights(self.model_path+"value_model_"+str(index)+".h5")

    def load_network(self, index):
        print("LOADING MODEL")
        ''' load model '''
        with open(self.model_path+"policy_mu_model_"+str(index)+".json") as policy_json:
            self.policy_mu_model = model_from_json(policy_json.read())
        with open(self.model_path+"value_model_"+str(index)+".json") as value_json:
            self.value_model = model_from_json(value_json.read())
        self.policy_mu_model.load_weights(self.model_path+"policy_mu_model_"+str(index)+".h5")
        self.value_model.load_weights(self.model_path+"value_model_"+str(index)+".h5")

        adam_optimizer = Adam(lr=self.value_learning_rate,
                              beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.value_model.compile(
            loss='mean_squared_error', optimizer=adam_optimizer)

        ''' model outputs '''
        self.mu = self.policy_mu_model(self.input_ph)
        self.value = self.value_model(self.input_ph)
        self.sigma = tf.compat.v1.get_variable('sigma', (1, self.env_action_number),
                                               tf.float32,
                                               tf.constant_initializer(0.6))

        ''' trainable weights '''
        self._mean_model_params = self.policy_mu_model.trainable_weights
        self._value_model_params = self.value_model.trainable_weights 
        
        print('\033[92m'+"POLICY MODEL STRUCTURE"+'\033[0m')
        self.policy_mu_model.summary()
        print('\033[92m'+"VALUE MODEL STRUCTURE"+'\033[0m')
        self.value_model.summary()
