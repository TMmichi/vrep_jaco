import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import numpy as np

import random
from datetime import datetime as dt

from state_gen.config import block_setting


#TODO: Make **kwargs fetch into the blocks
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
    

class FiLMedFCBlock(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.units = kwargs['units'] #Raise error when not defined
        self.activation = kwargs.get('activation','relu')
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self.debug = kwargs.get('debug',False)
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
        print(x.shape) if self.debug else None
        return x


class FushionBlock(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.units = kwargs['units'] #Raise error when not defined
        self.activation = kwargs.get('activation','relu')
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self.debug = kwargs.get('debug',False)
        self._build()

    def _build(self):
        #Input layer
        self.joint_input = tf.keras.layers.Input(shape=[6])
        self.gripper_input = tf.keras.layers.Input(shape=[3])
        self.pressure_input = tf.keras.layers.Input(shape=[12])
        #Concate layer
        self.concat = tf.keras.layers.concatenate()
        #FC layer
        self.fc = tf.keras.layers.Dense(units=self.units, activation=self.activation)
        #Dropout layer
        self.dropout = tf.keras.layers.Dropout(1-self.survival_prob)

     
    def call(self,
            inputs_imgfeat,
            inputs_joint,
            inputs_gripper,
            inputs_pressure,
            training = True):
        i = inputs_imgfeat
        j = self.joint_input(inputs_joint)
        g = self.gripper_input(inputs_gripper)
        p = self.pressure_input(inputs_pressure)
        
        x = self.concat([i,j,g,p])
        x = self.fc(x)
        x = self.dropout(x,training)
        print(x.shape) if self.debug else None
        return x


class FiLMedFushionBlock(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.units = kwargs['units'] #Raise error when not defined
        self.activation = kwargs.get('activation','relu')
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self.debug = kwargs.get('debug',False)
        self._build()

    def _build(self):
        #FILM layer
        self.film = FiLM()
        #Input layer
        self.joint_input = tf.keras.layers.Input(shape=[6])
        self.gripper_input = tf.keras.layers.Input(shape=[3])
        self.pressure_input = tf.keras.layers.Input(shape=[12])
        #Concate layer
        self.concat = tf.keras.layers.concatenate()
        #FC layer
        self.fc = tf.keras.layers.Dense(units=self.units, activation=self.activation)
        #Dropout layer
        self.dropout = tf.keras.layers.Dropout(1-self.survival_prob)

     
    def call(self,
            inputs_imgfeat,
            inputs_joint,
            inputs_gripper,
            inputs_pressure,
            gammas,
            betas,
            training = True):
        i = inputs_imgfeat
        j = self.joint_input(inputs_joint)
        g = self.gripper_input(inputs_gripper)
        p = self.pressure_input(inputs_pressure)
        
        x = self.concat([i,j,g,p])
        x = self.film(self.fc(x),gammas,betas)
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
        self.debug = kwargs.get('debug',False)
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
        print(x.shape) if self.debug else None
        return x


class fushion_Encoder(tf.keras.Model):
    def __init__(self,**kwargs):
        super().__init__()
        self.num_conv_blocks = kwargs.get('num_conv_blocks',5)
        self.num_fc_blocks = kwargs.get('num_fc_blocks',2)
        self.latent_dim = kwargs.get('latent_dim',32)
        self.condition_method = kwargs.get('condition_method','conv-film')
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self.drop_rate = 1.0 - self.survival_prob
        self.conv_isfilmed = kwargs.get('conv_isfilmed',False)
        self.fushion_isfilmed = kwargs.get('fushion_isfilmed',False)
        self.fc_isfilmed = kwargs.get('fc_isfilmed',False)
        self.latent_isfilmed = kwargs.get('latent_isfilmed',False)
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
            if self.conv_isfilmed:
                block = FiLMedConvBlock(**setting)
            else:
                block = ConvBlock(**setting)
            self._blocks.append([block,self.conv_isfilmed,block_id])
        #Flatten layer
        block = tf.keras.layers.Flatten()
        self._blocks.append([block,None,'flatten'])
        #Fushion layer
        block_id = 'fushion'
        setting = block_setting[block_id]
        if self.fushion_isfilmed:
            block = FiLMedFushionBlock(**setting)
            self._blocks.append([block,self.fushion_isfilmed,block_id])
        #FC layer
        for idx in range(self.num_fc_blocks):
            block_id = 'fc_block%s' % str(idx+1)
            print(block_id) if self.debug else None
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
            inputs_img,
            inputs_joint,
            inputs_gripper,
            inputs_pressure,
            gammas=None,
            betas=None,
            training=True):
        x = inputs_img
        j = inputs_joint
        g = inputs_gripper
        p = inputs_pressure
        for idx, [block,isfilm,block_id] in enumerate(self._blocks):
            #Conv Blocks (FiLMed or Not)
            with tf.compat.v1.variable_scope(block_id):
                if isfilm:
                    if gammas == None or betas == None:
                        raise NameError("Gammas, Betas are not given")
                    if block_id == 'fushion':
                        x = block(
                            inputs_imgfeat=x,
                            inputs_joint=j,
                            inputs_gripper=g,
                            inputs_pressure=p,
                            gammas=gammas,
                            betas=betas,
                            training=training)
                    else:
                        gammas = gammas[idx]
                        betas = betas[idx]
                        x = block(
                            x,
                            gammas=gammas,
                            betas=betas,
                            training=training)
                elif not isfilm:
                    if block_id == 'fushion':
                        x = block(
                            inputs_imgfeat=x,
                            inputs_joint=j,
                            inputs_gripper=g,
                            inputs_pressure=p,
                            training=training)
                    else:
                        x = block(
                            x, 
                            training=training)
                elif isfilm is None:
                    x = block(x)
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=1)
        return mean, logvar


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


class CNN_Encoder(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.num_conv_blocks = kwargs.get('num_conv_blocks',5)
        self.num_fc_blocks = kwargs.get('num_fc_blocks',2)
        self.latent_dim = kwargs.get('latent_dim',32)
        self.condition_method = kwargs.get('condition_method','conv-film')
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self.drop_rate = 1.0 - self.survival_prob
        self.conv_isfilmed = kwargs.get('conv_isfilmed',False)
        self.fc_isfilmed = kwargs.get('fc_isfilmed',False)
        self.latent_isfilmed = kwargs.get('latent_isfilmed',False)
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
            print(block_id) if self.debug else None
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
        # print('x time')
        # print(x)
        for idx, [block,isfilm,block_id] in enumerate(self._blocks):
            #Conv Blocks (FiLMed or Not)
            with tf.compat.v1.variable_scope(block_id):
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
                elif isfilm == False:
                    x = block(
                        x, 
                        training=training)
                elif isfilm is None:
                    x = block(x)
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=1)
        return mean, logvar


class CNN_Decoder(layers.Layer):
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
            print(block_id) if self.debug else None
            setting = block_setting[block_id]
            setting['survival_prob'] = 1.0 - self.drop_rate * float(idx) / len(self._blocks)
            block = DeConvBlock(**setting)
            self._blocks.append([block,False,block_id])
        '''
        #Deconv Output
        block_id = 'deconv_output'
        setting = block_setting[block_id]
        setting['survival_prob'] = 1.0 - self.drop_rate
        block = DeConvBlock(**setting)
        self._blocks.append([block,False,block_id])'''

    def call(self,
            z,
            gammas=None,
            betas=None,
            training=True,
            apply_sigmoid=False):
        x_hat = z
        for [block,isfilm,block_id] in self._blocks:
            #Conv Blocks (FiLMed or Not)
            with tf.compat.v1.variable_scope(block_id):
                if isfilm == False:
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
        self.latent_dim = kwargs.get('latent_dim',32)
        kwargs['num_deconv_blocks'] = kwargs.get('num_conv_blocks',5)
        kwargs['num_defc_blocks'] = kwargs.get('num_fc_blocks',2)
        self.isfushion = kwargs.get('isfushion') #Raise Error if not defined
        self.debug = kwargs.get('debug',False)
        self._build(**kwargs)

    def _build(self,**kwargs):
        if self.isfushion:
            self.fushionencoder = fushion_Encoder(**kwargs)
            self.decoder = CNN_Decoder(**kwargs)
        else:

            self.encoder = CNN_Encoder(**kwargs)
            self.decoder = CNN_Decoder(**kwargs)

            self.encoder1 = CNN_Encoder(**kwargs)
            self.decoder1 = CNN_Decoder(**kwargs)

            self.encoder2 = CNN_Encoder(**kwargs)
            self.decoder2 = CNN_Decoder(**kwargs)

            self.encoder3 = CNN_Encoder(**kwargs)
            self.decoder3 = CNN_Decoder(**kwargs)
            
    
    # @tf.function
    # def state(self,x):
    #     # mean,_ = self.encoder(x)
    #     mean,_ = self.encoder(x[0])
    #     mean1,_ = self.encoder1(x[1])
    #     mean2,_ = self.encoder2(x[2])
    #     mean3,_ = self.encoder3(x[3])
    #     return mean,mean1,mean2,mean3
    
    def reparameterize(self,mean,logvar):
        #eps = tf.random.normal(shape=mean.shape)
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean
    
    @tf.function
    def sample(self, sample_num=1,eps=None):
        if eps is None:
            #eps = tf.random.normal(shape=(sample_num, self.latent_dim))
            eps = tf.random.normal(shape=(sample_num, self.latent_dim))
        return self.decoder(eps, apply_sigmoid=True, training=False) # ,self.decoder1(eps, apply_sigmoid=True, training=False),self.decoder2(eps, apply_sigmoid=True, training=False),self.decoder3(eps, apply_sigmoid=True, training=False)

    def _log_normal_pdf(self,sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def compute_loss(self,x,x_hat):
        print('loss')
        print(x.shape)

        if self.isfushion:
            print("inputs shape in loss: ",x.shape)
            mean, logvar = self.fushionencoder(x[0],x[1],x[2],x[3])
        else:
            div, div1, div2, div3 = tf.split(x, num_or_size_splits=4)
            div_hat, div_hat1, div_hat2, div_hat3 = tf.split(x_hat, num_or_size_splits=4)
            # mean, logvar = self.encoder(x)
            # mean, logvar = self.encoder(x)
            mean, logvar = self.encoder(div)
            # mean1, logvar1 = self.encoder1(x)
            mean1, logvar1 = self.encoder1(div1)
            # mean2, logvar2 = self.encoder2(x)
            mean2, logvar2 = self.encoder2(div2)
            # mean3, logvar3 = self.encoder3(x)
            mean3, logvar3 = self.encoder3(div3)
        z = self.reparameterize(mean, logvar)
        z1 = self.reparameterize(mean1, logvar1)
        z2 = self.reparameterize(mean2, logvar2)
        z3 = self.reparameterize(mean3, logvar3)
        mean_f = tf.concat([mean,mean1,mean2,mean3],0)
        # x_logit = self.decoder(mean)
        # x_logit1 = self.decoder1(mean1)
        # x_logit2 = self.decoder2(mean2)
        # x_logit3 = self.decoder3(mean3)
        x_logit = self.decoder(mean_f)
        x_logit1 = self.decoder1(mean_f)
        x_logit2 = self.decoder2(mean_f)
        x_logit3 = self.decoder3(mean_f)
        # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=div_hat)
        cross_ent1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit1, labels=div_hat1)
        cross_ent2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit2, labels=div_hat2)
        cross_ent3 = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit3, labels=div_hat3)
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
    def compute_apply_gradients(self, x, optimizer):
        print('here????')
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x,x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    @tf.function
    def call(self,
            inputs,
            gammas=None,
            betas=None,
            training=True):
        print(inputs.get_shape())
        print("inputs shape in call: ",inputs[0].shape)
        print("inputs shape in call: ",inputs[1].shape)
        print("inputs shape in call: ",inputs[2].shape)
        print("inputs shape in call: ",inputs[3].shape)

        if self.isfushion:
            inputs_img = inputs[0]
            inputs_joint = inputs[1]
            inputs_gripper = inputs[2]
            inputs_pressure = inputs[3]
            mean, logvar = self.fushionencoder(inputs_img,
                                    inputs_joint=inputs_joint,
                                    inputs_gripper=inputs_gripper,
                                    inputs_pressure=inputs_pressure,
                                    gammas=None,
                                    betas=None,
                                    training=training)
            z = self.reparameterize(mean, logvar)
            probs = self.decoder(z)
        else:
            div, div1, div2, div3 = tf.split(inputs, num_or_size_splits=4, axis=1)
            div = tf.squeeze(div,[1])
            div1 = tf.squeeze(div1,[1])
            div2 = tf.squeeze(div2,[1])
            div3 = tf.squeeze(div3,[1])
            mean, logvar = self.encoder(div,gammas,betas)
            mean1, logvar1 = self.encoder1(div1,gammas,betas)
            mean2, logvar2 = self.encoder2(div2,gammas,betas)
            mean3, logvar3 = self.encoder3(div3,gammas,betas)
            z = self.reparameterize(mean, logvar)
            z1 = self.reparameterize(mean1, logvar1)
            z2 = self.reparameterize(mean2, logvar2)
            z3 = self.reparameterize(mean3, logvar3)
            z = tf.concat([z,z1,z2,z3],0)
            print(z)
            probs = self.decoder(z)
            print("hi")
        return probs

class dataFusionGraph(tf.keras.Model):
    def __init__(self):
        pass