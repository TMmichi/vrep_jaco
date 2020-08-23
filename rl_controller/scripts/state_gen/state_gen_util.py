import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import numpy as np

import random
from datetime import datetime as dt

from state_gen.config import block_setting

class FCBlock(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.units = kwargs['units']
        self.activation = kwargs.get('activation','relu')
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self.debug = kwargs.get('debug',False)
        self._build()
        
    def _build(self):
        self.fc = tf.keras.layers.Dense(units=self.units, activation=self.activation)
        self.dropout = tf.keras.layers.Dropout(1-self.survival_prob)
     
    def call(self,
            inputs,
            training = True):
        x = self.fc(inputs)
        x = self.dropout(x)
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
        x = self.dropout(x)
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
        x = self.dropout(x)
        print(x.shape) if self.debug else None
        return x

class DeConvBlock_fin(layers.Layer):
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
            # activation='relu')
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
        # if self.with_batch_norm:
        #     x = self.bn(x)
        # x = self.nonlin(x) if not self.nonlin is None else x
        # x = self.dropout(x)
        print(x.shape) if self.debug else None
        return x

class CNN_Encoder(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.num_conv_blocks = kwargs.get('num_conv_blocks',5)
        self.num_fc_blocks = kwargs.get('num_fc_blocks',2)
        self.latent_dim = kwargs.get('latent_dim',32)
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
            self._blocks.append([block,False,block_id])
        
        #Flatten layer
        block = tf.keras.layers.Flatten()
        self._blocks.append([block,None,'flatten'])
        #FC layer
        for idx in range(self.num_fc_blocks):
            block_id = 'fc_block%s' % str(idx+1)
            print(block_id) if self.debug else None
            setting = block_setting[block_id]
            block = FCBlock(**setting)
            self._blocks.append([block,False,block_id])
        #Latent Vector Encoder
        setting = block_setting['latent_layer']
        setting['units'] = 2 * self.latent_dim
        block = FCBlock(**setting)
        self._blocks.append([block,False,block_id])
     
    def call(self,
            inputs,
            gammas=None,
            betas=None,
            training=True):
        x = inputs
        for idx, [block,isfilm,block_id] in enumerate(self._blocks):
            #Conv Blocks (FiLMed or Not)
            with tf.compat.v1.variable_scope(block_id):
                if isfilm == False:
                    x = block(
                        x, 
                        training=training)
                elif isfilm is None:
                    x = block(x)
        return x

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
            
        setting = block_setting['deconv_output']
        setting['survival_prob'] = 1.0
        block = DeConvBlock_fin(**setting)
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
        
        
        # x_hat = tf.keras.activations.sigmoid(x_hat)
        return x_hat

class CNN_Encoder_latent(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.num_conv_blocks = kwargs.get('num_conv_blocks',5)
        self.num_fc_blocks = kwargs.get('num_fc_blocks',2)
        self.latent_dim2 = kwargs.get('latent_dim2',32)
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self.drop_rate = 1.0 - self.survival_prob
        self.debug = kwargs.get('debug',False)
        random.seed(dt.now())
        self.seed = random.randint(0,12345)
        self._build()
        
    def _build(self):
        # return mean, logvar
        #Building blocks
        self._blocks = []
        #FC layer
        for idx in range(self.num_fc_blocks):
            block_id = 'latent_fc_block%s' % str(idx+1)
            print(block_id) if self.debug else None
            setting = block_setting[block_id]
            setting['units'] = (2-idx) * 2 * self.latent_dim2
            block = FCBlock(**setting)
            self._blocks.append([block,False,block_id])
        block = FCBlock(**setting)
        self._blocks.append([block,False,block_id])
     
    def call(self,
            inputs,
            gammas=None,
            betas=None,
            training=True):
        x = inputs
        for idx, [block,isfilm,block_id] in enumerate(self._blocks):
            #Conv Blocks (FiLMed or Not)
            with tf.compat.v1.variable_scope(block_id):
                if isfilm == False:
                    x = block(
                        x, 
                        training=training)
                elif isfilm is None:
                    x = block(x)
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=1)
        return mean, logvar

class CNN_Decoder_latent(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.num_deconv_blocks = kwargs.get('num_deconv_blocks',5)
        self.num_defc_blocks = kwargs.get('num_defc_blocks',2)
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self.latent_dim2 = kwargs.get('latent_dim2',32)
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
            block_id = 'latent_defc_block%s' % str(idx+1)
            print(block_id) if self.debug else None
            setting = block_setting[block_id]
            setting['units'] = (2-idx) * 2 * self.latent_dim2
            block = FCBlock(**setting)
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

class Autoencoder_encoder():
    def __init__(self,**kwargs):
        self.debug = kwargs.get('debug',False)
        self._build(**kwargs)

    def _build(self,**kwargs):
        self.encoder = CNN_Encoder(**kwargs)
        self.encoder2 = CNN_Encoder(**kwargs)
        self.encoder3 = CNN_Encoder(**kwargs)
        self.encoder4 = CNN_Encoder(**kwargs)
        
        self.encoder_latent = CNN_Encoder_latent(**kwargs)

        self.input1 = tf.keras.Input(shape=(480,640,1))
        self.input2 = tf.keras.Input(shape=(480,640,1))
        self.input3 = tf.keras.Input(shape=(480,640,1))
        self.input4 = tf.keras.Input(shape=(480,640,1))

        self.en1 = self.encoder(self.input1)
        self.en2 = self.encoder2(self.input2)
        self.en3 = self.encoder3(self.input3)
        self.en4 = self.encoder4(self.input4)
        
        self.concate_latent = tf.keras.layers.Concatenate(axis=1)([self.en1, self.en2, self.en3, self.en4])
        
        self.en_latent = self.encoder_latent(self.concate_latent)
        
        self.encoder_model = tf.keras.Model([self.input1,self.input2,self.input3,self.input4],self.en_latent,name='encoder')
        
class Autoencoder_decoder():
    def __init__(self,**kwargs):
        self.debug = kwargs.get('debug',False)
        self._build(**kwargs)
        self.latent_dim = kwargs.get('latent_dim',32)
        self.latent_dim2 = kwargs.get('latent_dim2',32)

    def _build(self,**kwargs):
        
        self.inputs = tf.keras.Input(shape=(32,))
        
        self.decoder_latent = CNN_Decoder_latent(**kwargs)
                
        self.decoder1 = CNN_Decoder(**kwargs)
        self.decoder2 = CNN_Decoder(**kwargs)
        self.decoder3 = CNN_Decoder(**kwargs)
        self.decoder4 = CNN_Decoder(**kwargs)
        
        self.de_latent = self.decoder_latent(self.inputs)
        
        self.de1 = self.decoder1(self.de_latent)
        self.de2 = self.decoder2(self.de_latent)
        self.de3 = self.decoder3(self.de_latent)
        self.de4 = self.decoder4(self.de_latent)

        self.decoder_model1 = tf.keras.Model(self.inputs,self.de1,name='decoder1')
        self.decoder_model2 = tf.keras.Model(self.inputs,self.de2,name='decoder2')
        self.decoder_model3 = tf.keras.Model(self.inputs,self.de3,name='decoder3')
        self.decoder_model4 = tf.keras.Model(self.inputs,self.de4,name='decoder4')

class Autoencoder():
    def __init__(self,**kwargs):
        self.debug = kwargs.get('debug',False)
        
        self.input1 = tf.keras.Input(shape=(480,640,1),name='input1')
        self.input2 = tf.keras.Input(shape=(480,640,1),name='input2')
        self.input3 = tf.keras.Input(shape=(480,640,1),name='input3')
        self.input4 = tf.keras.Input(shape=(480,640,1),name='input4')
        
        self._build(**kwargs)
    
    def _build(self,**kwargs):
        self.encoder = Autoencoder_encoder()
        self.decoder = Autoencoder_decoder()
   
        self.en_mean, self.en_var = self.encoder.encoder_model([self.input1,self.input2,self.input3,self.input4])
        
        self.z = self.reparameterize(self.en_mean,self.en_var)

        self.logpz = self._log_normal_pdf(self.z,0.,0.)

        self.logqz_x = self._log_normal_pdf(self.z,self.en_mean,self.en_var)
        
        self.prob1 = tf.keras.activations.sigmoid(self.decoder.decoder_model1(self.z))
        self.prob2 = tf.keras.activations.sigmoid(self.decoder.decoder_model2(self.z))
        self.prob3 = tf.keras.activations.sigmoid(self.decoder.decoder_model3(self.z))
        self.prob4 = tf.keras.activations.sigmoid(self.decoder.decoder_model4(self.z))

        self.autoencoder = tf.keras.Model(inputs=[self.input1,self.input2,self.input3,self.input4],outputs=[self.prob1,self.prob2,self.prob3,self.prob4],name='multimodal_variational_autoencoder')

    def reparameterize(self,mean,logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def _log_normal_pdf(self,sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def kl_reconstruction_loss1(self, true, pred):
        # Reconstruction loss
        reconstruction_loss = tf.keras.losses.binary_crossentropy(K.flatten(true), K.flatten(pred))
        # reconstruction_loss = tf.keras.losses.MSE(K.flatten(true), K.flatten(pred))

        # KL divergence loss
        # mean , var = self.encoder.encoder_model1(true)
        # kl_loss = 1 + K.sum(var - K.square(mean) - K.exp(var),axis=-1)
        # kl_loss *= -0.5
        # Total loss = 50% rec + 50% KL divergence loss

        return K.mean(reconstruction_loss) # + kl_loss)
        # return 100 * K.mean(reconstruction_loss) # + kl_loss)

    def kl_reconstruction_loss2(self, true, pred):
        # Reconstruction loss
        reconstruction_loss = tf.keras.losses.binary_crossentropy(K.flatten(true), K.flatten(pred))
        # reconstruction_loss = tf.keras.losses.MSE(K.flatten(true), K.flatten(pred))

        # KL divergence loss
        # mean , var = self.encoder.encoder_model2(true)
        # kl_loss = 1 + K.sum(var - K.square(mean) - K.exp(var),axis=-1)
        # kl_loss *= -0.5
        # Total loss = 50% rec + 50% KL divergence loss

        return K.mean(reconstruction_loss) # + kl_loss)
        # return 100 * K.mean(reconstruction_loss) # + kl_loss)

    def kl_reconstruction_loss3(self, true, pred):
        # Reconstruction loss
        reconstruction_loss = tf.keras.losses.binary_crossentropy(K.flatten(true), K.flatten(pred))
        # reconstruction_loss = tf.keras.losses.MSE(K.flatten(true), K.flatten(pred))

        # KL divergence loss
        # mean , var = self.encoder.encoder_model3(true)
        # kl_loss = 1 + K.sum(var - K.square(mean) - K.exp(var),axis=-1)
        # kl_loss *= -0.5
        # Total loss = 50% rec + 50% KL divergence loss

        return K.mean(reconstruction_loss) # + kl_loss)
        # return 100 * K.mean(reconstruction_loss) # + kl_loss)

    def kl_reconstruction_loss4(self, true, pred):
        # Reconstruction loss
        reconstruction_loss = tf.keras.losses.binary_crossentropy(K.flatten(true), K.flatten(pred))
        # reconstruction_loss = tf.keras.losses.MSE(K.flatten(true), K.flatten(pred))

        # KL divergence loss
        # mean , var = self.encoder.encoder_model4(true)
        kl_loss = K.sum(1 + self.en_var - K.square(self.en_mean) - K.exp(self.en_var),axis=-1)
        kl_loss *= -0.5
        
        # Total loss = 50% rec + 50% KL divergence loss

        return K.mean(reconstruction_loss + kl_loss)
        # return K.mean(100 * reconstruction_loss + kl_loss)