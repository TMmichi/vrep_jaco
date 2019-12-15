import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import numpy as np

import random
from datetime import datetime as dt

from config import block_setting


class FiLM(layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self,x,gammas,betas):
        height = x.shape[1]
        width = x.shape[2]
        n_features = x.shape[-1]
        assert(gammas.shape[-1]==n_features)
        #gamma reformulation
        gammas = K.expand_dims(gammas,axis=1)
        gammas = K.expand_dims(gammas,axis=1)
        gammas = K.tile(gammas,[1,height,width,n_features])
        #betas reformulation
        betas = K.expand_dims(betas,axis=1)
        betas = K.expand_dims(betas,axis=1)
        betas = K.tile(betas,[1,height,width,n_features])

        return (gammas * x) + betas


class FCBlock(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.units = kwargs['units'] #Raise error when not defined
        self.activation = kwargs.get('activation','relu')
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self._build()
        
    def _build(self):
        #FC layer
        self.fc = tf.keras.layers.Dense(units=self.units, activation=self.activation)
        #Dropout layer
        self.dropout = tf.keras.layers.Dropout(1-self.survival_prob)

    def call(self,
            inputs,
            training = True):
        x = self.fc(inputs)
        x = self.dropout(x,training)
        print(x.shape)
        return x
    

class FiLMedFCBlock(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.z_dim = kwargs['units'] #Raise error when not defined
        self.activation = kwargs.get('activation','relu')
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self._build()

    def _build(self):
        #FILM layer
        self.film = FiLM()
        #FC layer
        self.fc = tf.keras.layers.Dense(units=self.units, activation=self.activation)
        #Dropout layer
        self.dropout = tf.keras.layers.Dropout(1-self.survival_prob)

    def call(self,
            inputs,
            gammas,
            betas,
            training = True):
        x = self.film(self.fc(inputs),gammas,betas)
        x = self.dropout(x,training)
        print(x.shape)
        return x


class ConvBlock(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.with_batch_norm = kwargs.get('with_batch_norm',True)
        if self.with_batch_norm:
            self._batch_norm_momentum = kwargs.get('batch_norm_momentum',0.99)
            self._batch_norm_epsilon = kwargs.get('batch_norm_epsilon',0.001)
        self.n_features = kwargs['n_features'] #Raise error when not defined
        self.kernel_size = kwargs.get('kernel_size',3)
        self.stride = kwargs.get('stride',(2,2))
        self.padding = kwargs.get('padding','SAME')
        self.activation = kwargs.get('activation','relu')
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self._build()

    def _build(self):
        #Conv layer
        self.conv = tf.keras.layers.Conv2D(
            filters=self.n_features, 
            kernel_size=self.kernel_size, 
            strides=self.stride,
            padding=self.padding)
        #BN layer
        if self.with_batch_norm:
            self.bn = tf.keras.layers.BatchNormalization(
                momentum=self._batch_norm_momentum, 
                epsilon=self._batch_norm_epsilon)
        #Non-linearity layer
        if self.activation == 'relu':
            self.nonlin = tf.keras.layers.ReLU()
        elif self.activation == 'elu':
            self.nonlin = tf.keras.layers.ELU()
        elif self.activation == 'Leakyrelu':
            self.nonlin = tf.keras.layers.LeakyReLU()
        elif self.activation == 'none':
            self.nonlin = None
        else:
            raise NotImplementedError("Unidentified Activation Layer")
        #Dropout layer
        self.dropout = tf.keras.layers.Dropout(1-self.survival_prob)
    
    def call(self,
            inputs,            
            training=True):
        x = inputs
        x = self.conv(x)
        if self.with_batch_norm:
            x = self.bn(x)
        x = self.nonlin(x) if not self.nonlin is None else x
        x = self.dropout(x,training)
        print(x.shape)
        return x


class FiLMedConvBlock(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__() 
        self.with_batch_norm = kwargs.get('with_batch_norm',True)
        if self.with_batch_norm:
            self._batch_norm_momentum = kwargs.get('batch_norm_momentum',0.99)
            self._batch_norm_epsilon = kwargs.get('batch_norm_epsilon',0.001)
        self.condition_method = kwargs.get('condition_method','conv-film')
        self.n_features = kwargs['n_features'] #Raise error when not defined
        self.kernel_size = kwargs.get('kernel_size',3)
        self.stride = kwargs.get('stride',(2,2))
        self.padding = kwargs.get('padding','SAME')
        self.activation = kwargs.get('activation','relu')
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self._build()
    
    def _build(self):
        #FILM layer
        self.film = FiLM()
        #Conv layer
        self.conv = tf.keras.layers.Conv2D(
            filters=self.n_features,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding=self.padding)
        #BN layer
        if self.with_batch_norm:
            self.bn = tf.keras.layers.BatchNormalization(
                momentum=self._batch_norm_momentum, 
                epsilon=self._batch_norm_epsilon)
        #Non-linearity layer
        if self.activation == 'relu':
            self.nonlin = tf.keras.layers.ReLU()
        elif self.activation == 'elu':
            self.nonlin = tf.keras.layers.ELU()
        elif self.activation == 'Leakyrelu':
            self.nonlin = tf.keras.layers.LeakyReLU()
        elif self.activation == 'none':
            self.nonlin = None
        else:
            raise NotImplementedError("Unidentified Activation Layer")
        #Dropout layer
        self.dropout = tf.keras.layers.Dropout(1-self.survival_prob)

    def call(self,
            inputs,
            gammas,
            betas,
            training=True):
        x = inputs
        if self.condition_method == 'input-film':
            x = self.film(x,gammas,betas)
        x = self.conv(x)
        if self.condition_method == 'conv-film':
            x = self.film(x,gammas,betas)
        if self.with_batch_norm:
            x = self.bn(x)
            if self.condition_method == 'bn-film':
                x = self.film(x,gammas,betas)
        elif not self.with_batch_norm and self.condition_method == 'bn-film':
            raise NameError('bn-film called without initializing BN layer. Should choose other option.')
        x = self.nonlin(x) if not self.nonlin is None else x
        if self.condition_method == 'relu-film':
            x = self.film(x,gammas,betas)
        x = self.dropout(x,training)
        print(x.shape)
        return x


class DeConvBlock(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.with_batch_norm = kwargs.get('with_batch_norm',True)
        if self.with_batch_norm:
            self._batch_norm_momentum = kwargs.get('batch_norm_momentum',0.99)
            self._batch_norm_epsilon = kwargs.get('batch_norm_epsilon',0.001)
        self.n_features = kwargs['n_features'] #Raise error when not defined
        self.kernel_size = kwargs.get('kernel_size',3)
        self.stride = kwargs.get('stride',(2,2))
        self.padding = kwargs.get('padding','SAME')
        self.activation = kwargs.get('activation','relu')
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self._build()

    def _build(self):
        #DeConv layer
        self.deconv = tf.keras.layers.Conv2DTranspose(
            filters=self.n_features, 
            kernel_size=self.kernel_size, 
            strides=self.stride,
            padding=self.padding)
        #BN layer
        if self.with_batch_norm:
            self.bn = tf.keras.layers.BatchNormalization(
                momentum=self._batch_norm_momentum, 
                epsilon=self._batch_norm_epsilon)
        #Non-linearity layer
        if self.activation == 'relu':
            self.nonlin = tf.keras.layers.ReLU()
        elif self.activation == 'elu':
            self.nonlin = tf.keras.layers.ELU()
        elif self.activation == 'Leakyrelu':
            self.nonlin = tf.keras.layers.LeakyReLU()
        elif self.activation == 'none':
            self.nonlin = None
        else:
            raise NotImplementedError("Unidentified Activation Layer")
        #Dropout layer
        self.dropout = tf.keras.layers.Dropout(1-self.survival_prob)
    
    def call(self,
            inputs,            
            training=True):
        x = inputs
        x = self.deconv(x)
        if self.with_batch_norm:
            x = self.bn(x)
        x = self.nonlin(x) if not self.nonlin is None else x
        x = self.dropout(x,training)
        print(x.shape)
        return x


class CNN_Encoder(tf.keras.Model):
    def __init__(self,**kwargs):
        super().__init__()
        self.num_conv_blocks = kwargs.get('num_conv_blocks',5)
        self.num_fc_blocks = kwargs.get('num_fc_blocks',2)
        self.latent_dim = kwargs.get('latent_dim',32)
        self.condition_method = kwargs.get('condition_method','conv-film')
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self.drop_rate = 1.0 - self.survival_prob
        self.d_switch = kwargs.get('d_switch',False)
        self.conv_isfilmed = kwargs.get('conv_isfilmed',False)
        self.fc_isfilmed = kwargs.get('fc_isfilmed',False)
        self.latent_isfilmed = kwargs.get('latent_isfilmed',False)
        random.seed(dt.now())
        self.seed = random.randint(0,12345)
        self._build()
        
    def _build(self):
        #Building blocks
        self._blocks = []
        #Stack Conv Blocks
        for idx in range(self.num_conv_blocks):
            block_id = 'conv_block%s' % str(idx+1)
            print(block_id)
            setting = block_setting[block_id]
            setting['survival_prob'] = 1.0 - self.drop_rate * float(idx) / self.num_conv_blocks
            if self.conv_isfilmed:
                block = FiLMedConvBlock(**setting)
            else:
                block = ConvBlock(**setting)
            self._blocks.append([block,self.conv_isfilmed,block_id])
        #Flatten layer
        block = tf.keras.layers.Flatten()
        self._blocks.append([block,None,'flatten'])
        #FC layer
        for idx in range(self.num_fc_blocks):
            block_id = 'fc_block%s' % str(idx+1)
            setting = block_setting[block_id]
            if self.fc_isfilmed:
                block = FiLMedFCBlock(**setting)
            else:
                block = FCBlock(**setting)
            self._blocks.append([block,self.fc_isfilmed,block_id])
        #Latent Vector Encoder
        setting = block_setting['latent_layer']
        setting['units'] = 2 * self.latent_dim
        if self.latent_isfilmed:
            block = FiLMedFCBlock(**setting)
        else:
            block = FCBlock(**setting)
        self._blocks.append([block,self.latent_isfilmed,block_id])

    def call(self,
            inputs,
            gammas=None,
            betas=None,
            training=True):
        x = inputs
        for idx, [block,isfilm,block_id] in enumerate(self._blocks):
            #Conv Blocks (FiLMed or Not)
            with tf.variable_scope(block_id):
                if isfilm:
                    if gammas == None or betas == None:
                        raise NameError("Gammas, Betas are not given")
                    gammas = gammas[idx]
                    betas = betas[idx]
                    x = block(
                        x,
                        gammas=gammas,
                        betas=betas,
                        training=training)
                elif not isfilm:
                    x = block(
                        x, 
                        training=training)
                elif isfilm is None:
                    x = block(x)
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=1)
        return mean, logvar


class CNN_Decoder(tf.keras.Model):
    def __init__(self,**kwargs):
        super().__init__()
        self.num_deconv_blocks = kwargs.get('num_deconv_blocks',5)
        self.num_defc_blocks = kwargs.get('num_defc_blocks',2)
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self.drop_rate = 1.0 - self.survival_prob
        self.d_switch = kwargs.get('d_switch',False)
        random.seed(dt.now())
        self.seed = random.randint(0,12345)
        self._build()
        
    def _build(self):
        #Building blocks
        self._blocks = []
        #DeFC layer
        for idx in range(self.num_defc_blocks):
            block_id = 'defc_block%s' % str(idx+1)
            setting = block_setting[block_id]
            block = FCBlock(**setting)
            self._blocks.append([block,False,block_id])     
        #Deflatten layer
        block_id = "reshape"
        setting = block_setting[block_id]
        block = FCBlock(**setting)
        self._blocks.append([block,False,block_id])
        block = tf.keras.layers.Reshape(target_shape=(15,20,128))
        self._blocks.append([block,None,"deflatten"])
        #Stack DeConv Blocks
        for idx in range(self.num_deconv_blocks):
            block_id = 'deconv_block%s' % str(idx+1)
            setting = block_setting[block_id]
            setting['survival_prob'] = 1.0 - self.drop_rate * float(idx) / len(self._blocks)
            block = DeConvBlock(**setting)
            self._blocks.append([block,False,block_id])
        #Deconv Output
        block_id = 'deconv_output'
        setting = block_setting[block_id]
        setting['survival_prob'] = 1.0 - self.drop_rate
        block = DeConvBlock(**setting)
        self._blocks.append([block,False,block_id])

    def call(self,
            z,
            gammas=None,
            betas=None,
            training=True,
            apply_sigmoid=False):
        x_hat = z
        for [block,isfilm,block_id] in self._blocks:
            #Conv Blocks (FiLMed or Not)
            with tf.variable_scope(block_id):
                if not isfilm:
                    x_hat = block(
                        x_hat, 
                        training=training)
                elif isfilm is None:
                    x_hat = block(x_hat)
        if apply_sigmoid:
            probs = tf.sigmoid(x_hat)
            return probs
        return x_hat


class Autoencoder(tf.keras.Model):
    def __init__(self,**kwargs):
        super().__init__()
        self.num_conv_blocks = kwargs.get('num_conv_blocks',5)
        self.num_fc_blocks = kwargs.get('num_fc_blocks',2)
        self.latent_dim = kwargs.get('latent_dim',32)
        self.condition_method = kwargs.get('condition_method','conv-film')
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self.drop_rate = 1.0 - self.survival_prob
        self.d_switch = kwargs.get('d_switch',False)
        self.conv_isfilmed = kwargs.get('conv_isfilmed',False)
        self.fc_isfilmed = kwargs.get('fc_isfilmed',False)
        self.latent_isfilmed = kwargs.get('latent_isfilmed',False)
        self._build()

    def _build(self):
        self.encoder = CNN_Encoder()
        self.decoder = CNN_Decoder()
    
    def reparam(self,mean,logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def sample(self, sample_num=1,eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(sample_num, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def call(self,
            inputs,
            gammas=None,
            betas=None,
            training=True):
        x = inputs
        mean, logvar = self.encoder(x,gammas,betas)
        z = self.reparam(mean, logvar)
        probs = self.decoder(z)
        return probs


'''
def CNN_Encoder(**kwargs):
    d_switch = kwargs['d_switch'] or kwargs['trainable']
    random.seed(dt.now())
    seed = random.randint(0,12345)
    x = kwargs['depth_input']
    with tf.compat.v1.variable_scope("CNN_Encoder"):
        print(x.shape)
        batch1 = tf.layers.batch_normalization(
            inputs=x, axis=-1, scale=True, training=trainable, name="BN1")
        layer1 = tf.layers.conv2d(
            inputs=batch1, filters=16, kernel_size=8, strides=(1,1), padding='same',
            activation=tf.nn.relu, trainable=trainable, name="Conv1")
        pool1 = tf.layers.max_pooling2d(
            inputs=layer1,pool_size=[2,2],strides=2)
        dropout1 = tf.layers.dropout(
            inputs=pool1, rate=drop_rate, seed=seed, training=d_switch)
        print(dropout1.shape)

        batch2 = tf.layers.batch_normalization(
            inputs=dropout1, axis=-1, scale=True, training=trainable, name="BN2")
        layer2 = tf.layers.conv2d(
            inputs=batch2, filters=32, kernel_size=4, strides=(1,1), padding='same',
            activation=tf.nn.relu, trainable=trainable, name="Conv2")
        pool2 = tf.layers.max_pooling2d(
            inputs=layer2,pool_size=[2,2],strides=2)
        dropout2 = tf.layers.dropout(
            inputs=pool2, rate=drop_rate, seed=seed, training=d_switch)
        print(dropout2.shape)

        batch3 = tf.layers.batch_normalization(
            inputs=dropout2, axis=-1, scale=True, training=trainable, name="BN3")
        layer3 = tf.layers.conv2d(
            inputs=batch3, filters=64, kernel_size=4, strides=(1,1), padding='same',
            activation=tf.nn.relu, trainable=trainable, name="Conv3")
        pool3 = tf.layers.max_pooling2d(
            inputs=layer3,pool_size=[2,2],strides=2)
        dropout3 = tf.layers.dropout(
            inputs=pool3, rate=drop_rate, seed=seed, training=d_switch)
        print(dropout3.shape)

        batch4 = tf.layers.batch_normalization(
            inputs=dropout3, axis=-1, scale=True, training=trainable, name="BN4")
        layer4 = tf.layers.conv2d(
            inputs=batch4, filters=128, kernel_size=4, strides=(1,1), padding='same',
            activation=tf.nn.relu, trainable=trainable, name="Conv4")
        pool4 = tf.layers.max_pooling2d(
            inputs=layer4,pool_size=[2,2],strides=2)
        dropout4 = tf.layers.dropout(
            inputs=pool4, rate=drop_rate, seed=seed, training=d_switch)
        print(dropout4.shape)

        flat = tf.layers.flatten(dropout4)
        print(flat.shape)
        fc1 = tf.layers.dense(
            inputs=flat, units=512, name="FC1")
        dropout_fc1 = tf.layers.dropout(
            inputs=fc1, rate=drop_rate, seed=seed, training=d_switch)
        print(dropout_fc1.shape)
        
        fc2 = tf.layers.dense(
            inputs=dropout_fc1, units=2*z_dim, name="FC2")
        print(fc2.shape)
        
        mean = fc2[:, :z_dim]
        stddev = 1e-6 + tf.nn.softplus(fc2[:, z_dim:])

    return mean, stddev


def CNN_Decoder(z, drop_rate=0.2, reuse=False):
    with tf.variable_scope("CNN_Decoder", reuse=reuse):
        de_fc1 = tf.layers.dense(
            inputs=z, units=512, name="de_FC1")
        dropout_de_fc1 = tf.layers.dropout(
            inputs=de_fc1, rate=drop_rate)

        de_fc2 = tf.layers.dense(
            inputs=dropout_de_fc1, units=153600, name="de_FC2")
        dropout_de_fc2 = tf.layers.dropout(
            inputs=de_fc2, rate=drop_rate)

        unflat = tf.reshape(
            tensor=dropout_de_fc2, shape=[-1,30,40,128])

        de_layer1 = tf.layers.conv2d_transpose(
            inputs=unflat, filters=64, kernel_size=5, strides=(2,2), padding='same',
            activation=tf.nn.relu, name="DeConv1")
        de_dropout1 = tf.layers.dropout(
            inputs=de_layer1, rate=drop_rate)

        de_layer2 = tf.layers.conv2d_transpose(
            inputs=de_dropout1, filters=32, kernel_size=5, strides=(2,2), padding='same',
            activation=tf.nn.relu, name="DeConv2")
        de_dropout2 = tf.layers.dropout(
            inputs=de_layer2, rate=drop_rate) 

        de_layer3 = tf.layers.conv2d_transpose(
            inputs=de_dropout2, filters=16, kernel_size=6, strides=(2,2), padding='same',
            activation=tf.nn.relu, name="DeConv3")
        de_dropout3 = tf.layers.dropout(
            inputs=de_layer3, rate=drop_rate)
        
        de_layer4 = tf.layers.conv2d_transpose(
            inputs=de_dropout3, filters=1, kernel_size=8, strides=(2,2), padding='same', name="DeConv4")
        
        x_hat = tf.sigmoid(de_layer4)

    return x_hat


# Gateway
def autoencoder(x, dim_z, drop_rate=0.2, trainable=True):

    # encoding
    mu, sigma = CNN_Encoder(x, dim_z, drop_rate, trainable)

    # sampling by re-parameterization technique
    z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    # decoding
    x_hat = CNN_Decoder(z, drop_rate)
    x_hat = tf.clip_by_value(x_hat, 1e-8, 1 - 1e-8)

    # loss
    marginal_likelihood = tf.reduce_sum(x * tf.log(1e-8 +x_hat) + (1 - x) * tf.log(1e-8 + 1 - x_hat), 1)
    #marginal_likelihood = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x,logits=x_hat))
    KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    KL_divergence = tf.reduce_mean(KL_divergence)

    ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO

    return x_hat, z, loss, -marginal_likelihood, KL_divergence


def feature_fushion_MLP(mean_feature, input_placeholder):

    return []

#TODO: build data_fusion graph
def data_fusion_graph(input_placeholder):
    mean_feature = []
    # depth_arm
    mean_feature.append(CNN_Encoder(input_placeholder[0], z_dim=64, trainable=False)[0])
    # depth_bed
    mean_feature.append(CNN_Encoder(input_placeholder[1], z_dim=64, trainable=False)[0])
    # image_arm
    mean_feature.append(CNN_Encoder(input_placeholder[2], z_dim=64, trainable=False)[0])
    # image_bed
    mean_feature.append(CNN_Encoder(input_placeholder[3], z_dim=64, trainable=False)[0])

    state = feature_fushion_MLP(mean_feature, input_placeholder)

    return state

'''