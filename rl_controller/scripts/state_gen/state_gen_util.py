import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import numpy as np
import random
from datetime import datetime as dt
from state_gen.config import block_setting



############################################################################
########################   basic block gernerator   ########################
############################################################################
class FCBlock(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.units = kwargs['units']
        self.activation = kwargs.get('activation','relu')
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self.debug = kwargs.get('debug',False)
        self.is_dropout = kwargs.get('is_dropout',True)
        self._build()
        
    def _build(self):
        self.fc = tf.keras.layers.Dense(units=self.units, activation=self.activation, )
        self.dropout = tf.keras.layers.Dropout(1-self.survival_prob)
     
    def call(self,
            inputs):
        x = self.fc(inputs)
        if self.is_dropout:
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
        self.is_dropout = kwargs.get('is_dropout',True)
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
        if self.is_dropout:
            self.dropout = tf.keras.layers.Dropout(1-self.survival_prob)
    
     
    def call(self,
            inputs):
        x = inputs
        x = self.conv(x)
        if self.with_batch_norm:
            x = self.bn(x)
        x = self.nonlin(x) if not self.nonlin is None else x
        if self.is_dropout:
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
        self.is_dropout = kwargs.get('is_dropout',True)
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
            inputs):
        x = inputs
        x = self.deconv(x)
        if self.with_batch_norm:
            x = self.bn(x)
        x = self.nonlin(x) if not self.nonlin is None else x
        if self.is_dropout:
            x = self.dropout(x)
        print(x.shape) if self.debug else None
        return x



############################################################################
########################   middle block gernerator   #######################
############################################################################

class CNN_Encoder(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.num_conv_blocks = kwargs.get('num_conv_blocks',5)
        self.num_fc_bm_blocks = kwargs.get('num_fc_bm_blocks',2)
        self.num_fc_am_blocks = kwargs.get('num_fc_am_blocks',2)
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
        for idx in range(self.num_conv_blocks-1):
            block_id = 'conv_block%s' % str(idx+1)
            print(block_id) if self.debug else None
            setting = block_setting[block_id]
            setting['survival_prob'] = 1.0 - self.drop_rate * float(idx) / self.num_conv_blocks
            block = ConvBlock(**setting)
            self._blocks.append([block,block_id])
        
        block_id = 'conv_block_final'
        print(block_id) if self.debug else None
        setting = block_setting[block_id]
        setting['survival_prob'] = 1.0 - self.drop_rate * float(idx) / self.num_conv_blocks
        block = ConvBlock(**setting)
        self._blocks.append([block,block_id])

        #Flatten layer
        block_id = 'flatten'
        block = tf.keras.layers.Flatten()
        self._blocks.append([block,block_id])

        #FC layer
        for idx in range(self.num_fc_bm_blocks):
            block_id = 'conv_fc_bm_%s' % str(idx+1)
            print(block_id) if self.debug else None
            setting = block_setting[block_id]
            block = FCBlock(**setting)
            self._blocks.append([block,block_id])
                 
    def call(self,
            inputs,
            gammas=None,
            betas=None):
        x = inputs
        for idx, [block,block_id] in enumerate(self._blocks):
            x = block(x)
        return x

class CNN_Encoder_latent(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.num_conv_blocks = kwargs.get('num_conv_blocks',5)
        self.num_fc_bm_blocks = kwargs.get('num_fc_bm_blocks',2)
        self.num_fc_am_blocks = kwargs.get('num_fc_am_blocks',2)
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self.debug = kwargs.get('debug',False)
        self.drop_rate = 1.0 - self.survival_prob
        random.seed(dt.now())
        self.seed = random.randint(0,12345)
        self._build()
        
    def _build(self):
        #Building blocks
        self._blocks = []
        
        #FC layer
        for idx in range(self.num_fc_am_blocks):
            block_id = 'conv_fc_am_%s' % str(idx+1)
            print(block_id) if self.debug else None
            setting = block_setting[block_id]
            block = FCBlock(**setting)
            self._blocks.append([block,block_id])
                 
    def call(self,
            inputs,
            gammas=None,
            betas=None):
        x = inputs
        for idx, [block,block_id] in enumerate(self._blocks):
            x = block(x)
        mean_, logvar_ = tf.split(x, num_or_size_splits=2, axis=1, name='split')
        return mean_, logvar_

class CNN_Decoder_latent(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.num_conv_blocks = kwargs.get('num_deconv_blocks',5)
        self.num_fc_bd_blocks = kwargs.get('num_fc_bd_blocks',2)
        self.num_fc_ad_blocks = kwargs.get('num_fc_ad_blocks',2)
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self.drop_rate = 1.0 - self.survival_prob
        self.debug = kwargs.get('debug',False)
        random.seed(dt.now())
        self.seed = random.randint(0,12345)
        self._build()
        
    def _build(self):
        #Building blocks
        self._blocks = []
        
        #FC layer
        for idx in range(self.num_fc_bd_blocks):
            block_id = 'deconv_fc_bd_%s' % str(idx+1)
            print(block_id) if self.debug else None
            setting = block_setting[block_id]
            block = FCBlock(**setting)
            self._blocks.append([block,block_id])
                 
    def call(self,
            inputs,
            gammas=None,
            betas=None):
        x = inputs
        for idx, [block,block_id] in enumerate(self._blocks):
            x = block(x)
        return x

class CNN_Decoder(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.num_deconv_blocks = kwargs.get('num_deconv_blocks',5)
        self.num_fc_bd_blocks = kwargs.get('num_fc_bd_blocks',2)
        self.num_fc_ad_blocks = kwargs.get('num_fc_ad_blocks',2)
        self.survival_prob = kwargs.get('survival_prob',0.8)
        self.apply_sigmoid = kwargs.get('apply_sigmoid',True)
        self.drop_rate = 1.0 - self.survival_prob
        self.debug = kwargs.get('debug',False)

        random.seed(dt.now())
        self.seed = random.randint(0,12345)
        self._build()
        
    def _build(self):
        #Building blocks
        self._blocks = []
        #DeFC layer
        for idx in range(self.num_fc_ad_blocks):
            block_id = 'deconv_fc_ad_%s' % str(idx+1)
            print(block_id) if self.debug else None
            setting = block_setting[block_id]
            block = FCBlock(**setting)
            self._blocks.append([block,block_id])   
        #Reshape Tensor    
        block_id = 'reshape'
        block = tf.keras.layers.Reshape(target_shape=(15,20,128))
        self._blocks.append([block,block_id])
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
            betas=None):
        x_hat = z
        for [block,block_id] in self._blocks:
            x_hat = block(x_hat)
        if self.apply_sigmoid:
            probs = tf.sigmoid(x_hat)
            return probs
        return x_hat



##########################################################################
########################   full model gernerator   #######################
##########################################################################
class VAE_model(tf.keras.Model):
    def __init__(self,**kwargs):
        super(VAE_model, self).__init__()
        self.debug = kwargs.get('debug',False)
        self._build(**kwargs)
    
    def _build(self,**kwargs):
        #Build FULL VAE model frame
        self.encoder = CNN_Encoder(**kwargs)
        self.encoder_latent = CNN_Encoder_latent(**kwargs)
        self.decoder_latent = CNN_Decoder_latent(**kwargs)
        self.decoder = CNN_Decoder(**kwargs)

    def call(self, inputs):
        self.mean, self.logvar = self.encoder_latent(self.encoder(inputs))
        self.z = self.reparameterize(self.mean,self.logvar)
        self.prob = self.decoder(self.decoder_latent(self.z))
        return self.prob
    
    def reparameterize(self,mean,logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def vae_loss(self, true, pred):
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(true, pred)
        )
        kl_loss = 1 + self.logvar - tf.square(self.mean) - tf.exp(self.logvar)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss
        return total_loss





if __name__=="__main__":
    sample = tf.keras.Input(shape=(480,640,1),name='input')
    model = VAE_model(debug=True)
    result = model(sample)
    model_f = tf.keras.Model(sample,result)
    print(model_f.summary())