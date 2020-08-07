import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import numpy as np

import random
from datetime import datetime as dt

'''
    To fix layer you should change value in config and here kwargs
'''

from state_gen.config import block_setting

class FCBlock(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.units = kwargs['units'] #Raise error when not defined
        self.activation = kwargs.get('activation','relu')
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self.debug = kwargs.get('debug',False)
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
        print(x.shape) if self.debug else None
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
        self.debug = kwargs.get('debug',False)
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
        print(x.shape) if self.debug else None
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
        self.debug = kwargs.get('debug',False)
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
        print(x.shape) if self.debug else None
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
        self.debug = kwargs.get('debug',False)
        random.seed(dt.now())
        self.seed = random.randint(0,12345)
        self._build()
        
    def _build(self):
        #Building blocks
        self._blocks = []
        #Stack Conv Blocks
        for idx in range(self.num_conv_blocks):
            block_id = 'conv_block%s' % str(idx+1)
            print(block_id) if self.debug else None
            setting = block_setting[block_id]
            setting['survival_prob'] = 1.0 - self.drop_rate * float(idx) / self.num_conv_blocks
            block = ConvBlock(**setting)
            self._blocks.append([block,block_id])
        #Flatten layer
        block = tf.keras.layers.Flatten()
        self._blocks.append([block,None,'flatten'])
        #FC layer
        for idx in range(self.num_fc_blocks):
            block_id = 'fc_block%s' % str(idx+1)
            print(block_id) if self.debug else None
            setting = block_setting[block_id]
            block = FCBlock(**setting)
            self._blocks.append([block,block_id])
        #Latent Vector Encoder
        setting = block_setting['latent_layer']
        setting['units'] = 2 * self.latent_dim
        block = FCBlock(**setting)
        self._blocks.append([block,block_id])
     
    def call(self,
            inputs,
            gammas=None,
            betas=None,
            training=True):
        x = inputs
        # print('x time')
        # print(x)
        for idx, [block,block_id] in enumerate(self._blocks):
            #Conv Blocks (FiLMed or Not)
            with tf.compat.v1.variable_scope(block_id):
                x = block(
                    x, 
                    training=training)
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=1)
        return mean, logvar

class CNN_Decoder(tf.keras.Model):
    def __init__(self,**kwargs):
        super().__init__()
        self.num_deconv_blocks = kwargs.get('num_deconv_blocks',5)
        self.num_defc_blocks = kwargs.get('num_defc_blocks',2)
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self.drop_rate = 1.0 - self.survival_prob
        self.debug = kwargs.get('debug',False)

        random.seed(dt.now())
        self.seed = random.randint(0,12345)
        self._build()
        
    def _build(self):
        #Building blocks
        self._blocks = []
        #DeFC layer
        for idx in range(self.num_defc_blocks):
            block_id = 'defc_block%s' % str(idx+1)
            print(block_id) if self.debug else None
            setting = block_setting[block_id]
            block = FCBlock(**setting)
            self._blocks.append([block,block_id])     
        #Deflatten layer
        block_id = "reshape"
        setting = block_setting[block_id]
        block = FCBlock(**setting)
        self._blocks.append([block,block_id])
        block = tf.keras.layers.Reshape(target_shape=(15,20,128))
        self._blocks.append([block,"deflatten"])
        #Stack DeConv Blocks
        for idx in range(self.num_deconv_blocks):
            block_id = 'deconv_block%s' % str(idx+1)
            print(block_id) if self.debug else None
            setting = block_setting[block_id]
            setting['survival_prob'] = 1.0 - self.drop_rate * float(idx) / len(self._blocks)
            block = DeConvBlock(**setting)
            self._blocks.append([block,block_id])

    def call(self,
            z,
            gammas=None,
            betas=None,
            training=True,
            apply_sigmoid=False):
        x_hat = z
        for [block,block_id] in self._blocks:
            #Conv Blocks (FiLMed or Not)
            with tf.compat.v1.variable_scope(block_id):
                x_hat = block(
                    x_hat, 
                    training=training)
        if apply_sigmoid:
            probs = tf.sigmoid(x_hat)
            return probs
        return x_hat

class Autoencoder(tf.keras.Model):
    def __init__(self,**kwargs):
        super().__init__()
        self.latent_dim = kwargs.get('latent_dim',32)
        kwargs['num_deconv_blocks'] = kwargs.get('num_conv_blocks',5)
        kwargs['num_defc_blocks'] = kwargs.get('num_fc_blocks',2)
        self.debug = kwargs.get('debug',False)
        self._build(**kwargs)

    def _build(self,**kwargs):
        self.encoder = CNN_Encoder(**kwargs)
        self.decoder = CNN_Decoder(**kwargs)

        self.encoder1 = CNN_Encoder(**kwargs)
        self.decoder1 = CNN_Decoder(**kwargs)

        self.encoder2 = CNN_Encoder(**kwargs)
        self.decoder2 = CNN_Decoder(**kwargs)

        self.encoder3 = CNN_Encoder(**kwargs)
        self.decoder3 = CNN_Decoder(**kwargs)
                
    def reparameterize(self,mean,logvar):
        #eps = tf.random.normal(shape=mean.shape)
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean
    
    def _log_normal_pdf(self,sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def compute_loss(self,x,x_hat):
        print('loss')
        print(x.shape)
        print('loss2')

        # mean, logvar = self.encoder(x)
        mean, logvar = self.encoder(x[0])
        mean1, logvar1 = self.encoder1(x[1])
        mean2, logvar2 = self.encoder2(x[2])
        mean3, logvar3 = self.encoder3(x[3])

        z = self.reparameterize(mean, logvar)
        z1 = self.reparameterize(mean1, logvar1)
        z2 = self.reparameterize(mean2, logvar2)
        z3 = self.reparameterize(mean3, logvar3)
        mean_f = tf.concat([mean,mean1,mean2,mean3],0)

        x_logit = self.decoder(mean_f)
        x_logit1 = self.decoder1(mean_f)
        x_logit2 = self.decoder2(mean_f)
        x_logit3 = self.decoder3(mean_f)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x[0])
        cross_ent1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit1, labels=x[1])
        cross_ent2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit2, labels=x[2])
        cross_ent3 = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit3, labels=x[3])

        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpx_z1 = -tf.reduce_sum(cross_ent1, axis=[1, 2, 3])
        logpx_z2 = -tf.reduce_sum(cross_ent2, axis=[1, 2, 3])
        logpx_z3 = -tf.reduce_sum(cross_ent3, axis=[1, 2, 3])
        
        logpz = self._log_normal_pdf(z, 0., 0.)
        logpz1 = self._log_normal_pdf(z1, 0., 0.)
        logpz2 = self._log_normal_pdf(z2, 0., 0.)
        logpz3 = self._log_normal_pdf(z3, 0., 0.)
        
        logqz_x = self._log_normal_pdf(z, mean, logvar)
        logqz_x1 = self._log_normal_pdf(z1, mean1, logvar1)
        logqz_x2 = self._log_normal_pdf(z2, mean2, logvar2)
        logqz_x3 = self._log_normal_pdf(z3, mean3, logvar3)
        
        result = -tf.reduce_mean(logpx_z+logpx_z1+logpx_z2+logpx_z3 + logpz+logpz1+logpz2+logpz3 - logqz_x-logqz_x1-logqz_x2-logqz_x3)
        
        return result
    
    @tf.function
    def call(self,
            inputs,
            gammas=None,
            betas=None,
            training=True):
        print("inputs shape in call: ",inputs.shape)
        input_1, input_2, input_3, input_4 = tf.split(inputs, [480, 480, 480, 480], 1)
        
        mean, logvar = self.encoder(input_1,gammas,betas)
        mean1, logvar1 = self.encoder1(input_2,gammas,betas)
        mean2, logvar2 = self.encoder2(input_3,gammas,betas)
        mean3, logvar3 = self.encoder3(input_4,gammas,betas)

        z = self.reparameterize(mean, logvar)
        z1 = self.reparameterize(mean1, logvar1)
        z2 = self.reparameterize(mean2, logvar2)
        z3 = self.reparameterize(mean3, logvar3)

        z_f = tf.concat([z,z1,z2,z3],0)

        probs = self.decoder(z_f)
        probs1 = self.decoder1(z_f)
        probs2 = self.decoder2(z_f)
        probs3 = self.decoder3(z_f)

        probs = tf.concat([probs,probs1,probs2,probs3],0)       

        return probs
