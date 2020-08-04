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
        x = self.dropout(x)
        # x = self.dropout(x,training)

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
        # x = self.dropout(x,training)
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
        # x = self.dropout(x,training)

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
        # self.encoder2 = CNN_Encoder(**kwargs)
        # self.encoder3 = CNN_Encoder(**kwargs)
        # self.encoder4 = CNN_Encoder(**kwargs)

        self.input1 = tf.keras.Input(shape=(480,640,1))
        self.input2 = tf.keras.Input(shape=(480,640,1))
        self.input3 = tf.keras.Input(shape=(480,640,1))
        self.input4 = tf.keras.Input(shape=(480,640,1))

        self.en1 = self.encoder(self.input1)
        self.en2 = self.encoder(self.input2)
        self.en3 = self.encoder(self.input3)
        self.en4 = self.encoder(self.input4)

        self.encoder_model1 = tf.keras.Model(self.input1,self.en1,name='encoder1')
        self.encoder_model2 = tf.keras.Model(self.input2,self.en2,name='encoder2')
        self.encoder_model3 = tf.keras.Model(self.input3,self.en3,name='encoder3')
        self.encoder_model4 = tf.keras.Model(self.input4,self.en4,name='encoder4')

class Autoencoder_decoder():
    def __init__(self,**kwargs):
        self.debug = kwargs.get('debug',False)
        self._build(**kwargs)

    def _build(self,**kwargs):
        self.decoder = CNN_Decoder(**kwargs)
        # self.decoder2 = CNN_Decoder(**kwargs)
        # self.decoder3 = CNN_Decoder(**kwargs)
        # self.decoder4 = CNN_Decoder(**kwargs)

        self.input1 = tf.keras.Input(shape=(32,))
        self.input2 = tf.keras.Input(shape=(32,))
        self.input3 = tf.keras.Input(shape=(32,))
        self.input4 = tf.keras.Input(shape=(32,))

        self.inputs = tf.concat([self.input1,self.input2,self.input3,self.input4],1)
        self.de1 = self.decoder(self.inputs)
        self.de2 = self.decoder(self.inputs)
        self.de3 = self.decoder(self.inputs)
        self.de4 = self.decoder(self.inputs)

        self.decoder_model1 = tf.keras.Model([self.input1,self.input2,self.input3,self.input4],self.de1,name='decoder1')
        self.decoder_model2 = tf.keras.Model([self.input1,self.input2,self.input3,self.input4],self.de2,name='decoder2')
        self.decoder_model3 = tf.keras.Model([self.input1,self.input2,self.input3,self.input4],self.de3,name='decoder3')
        self.decoder_model4 = tf.keras.Model([self.input1,self.input2,self.input3,self.input4],self.de4,name='decoder4')


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
   
        self.en1_mean, self.en1_var = self.encoder.encoder_model1(self.input1)
        self.en2_mean, self.en2_var = self.encoder.encoder_model2(self.input2)
        self.en3_mean, self.en3_var = self.encoder.encoder_model3(self.input3)
        self.en4_mean, self.en4_var = self.encoder.encoder_model4(self.input4)
   
        self.z1 = self.reparameterize(self.en1_mean,self.en1_var)
        self.z2 = self.reparameterize(self.en2_mean,self.en2_var)
        self.z3 = self.reparameterize(self.en3_mean,self.en3_var)
        self.z4 = self.reparameterize(self.en4_mean,self.en4_var)

        self.logpz1 = self._log_normal_pdf(self.z1,0.,0.)
        self.logpz2 = self._log_normal_pdf(self.z2,0.,0.)
        self.logpz3 = self._log_normal_pdf(self.z3,0.,0.)
        self.logpz4 = self._log_normal_pdf(self.z4,0.,0.)

        self.logqz_x1 = self._log_normal_pdf(self.z1,self.en1_mean,self.en1_var)
        self.logqz_x2 = self._log_normal_pdf(self.z2,self.en2_mean,self.en2_var)
        self.logqz_x3 = self._log_normal_pdf(self.z3,self.en3_mean,self.en3_var)
        self.logqz_x4 = self._log_normal_pdf(self.z4,self.en4_mean,self.en4_var)
        
        self.prob1 = self.decoder.decoder_model1([self.z1,self.z2,self.z3,self.z4])
        self.prob2 = self.decoder.decoder_model2([self.z1,self.z2,self.z3,self.z4])
        self.prob3 = self.decoder.decoder_model3([self.z1,self.z2,self.z3,self.z4])
        self.prob4 = self.decoder.decoder_model4([self.z1,self.z2,self.z3,self.z4])

        self.autoencoder = tf.keras.Model(inputs=[self.input1,self.input2,self.input3,self.input4],outputs=[self.prob1,self.prob2,self.prob3,self.prob4],name='multimodal_variational_autoencoder')

    def reparameterize(self,mean,logvar):
        #eps = tf.random.normal(shape=mean.shape)
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def _log_normal_pdf(self,sample, mean, logvar, raxis=1):
        print('why??')
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def kl_reconstruction_loss1(self, true, pred):
        # Reconstruction loss
        reconstruction_loss = tf.keras.losses.binary_crossentropy(K.flatten(true), K.flatten(pred))
        # KL divergence loss
        mean , var = self.encoder.encoder_model1(true)
        kl_loss = 1 + K.sum(var - K.square(mean) - K.exp(var),axis=-1)
        kl_loss *= -0.5
        # Total loss = 50% rec + 50% KL divergence loss
        return K.mean(reconstruction_loss + kl_loss)

    def kl_reconstruction_loss2(self, true, pred):
        # Reconstruction loss
        reconstruction_loss = tf.keras.losses.binary_crossentropy(K.flatten(true), K.flatten(pred))
        # KL divergence loss
        mean , var = self.encoder.encoder_model2(true)
        kl_loss = 1 + K.sum(var - K.square(mean) - K.exp(var),axis=-1)
        kl_loss *= -0.5
        # Total loss = 50% rec + 50% KL divergence loss
        return K.mean(reconstruction_loss + kl_loss)

    def kl_reconstruction_loss3(self, true, pred):
        # Reconstruction loss
        reconstruction_loss = tf.keras.losses.binary_crossentropy(K.flatten(true), K.flatten(pred))
        # KL divergence loss
        mean , var = self.encoder.encoder_model3(true)
        kl_loss = 1 + K.sum(var - K.square(mean) - K.exp(var),axis=-1)
        kl_loss *= -0.5
        # Total loss = 50% rec + 50% KL divergence loss
        return K.mean(reconstruction_loss + kl_loss)

    def kl_reconstruction_loss4(self, true, pred):
        # Reconstruction loss
        reconstruction_loss = tf.keras.losses.binary_crossentropy(K.flatten(true), K.flatten(pred))
        # KL divergence loss
        mean , var = self.encoder.encoder_model4(true)
        kl_loss = 1 + K.sum(var - K.square(mean) - K.exp(var),axis=-1)
        kl_loss *= -0.5
        # Total loss = 50% rec + 50% KL divergence loss
        return K.mean(reconstruction_loss + kl_loss)
    
    def compute_loss(self,x,x_hat):
        # print(x_hat.shape)
        # # quit()
        # div, div1, div2, div3 = tf.split(x, num_or_size_splits=4,axis=1)
        # div_hat, div_hat1, div_hat2, div_hat3 = tf.split(x_hat, num_or_size_splits=4)
        # if len(div.shape)==5:
        #     div = tf.squeeze(div,[1])
        # if len(div1.shape)==5:
        #     div1 = tf.squeeze(div1,[1])
        # if len(div2.shape)==5:
        #     div2 = tf.squeeze(div2,[1])
        # if len(div3.shape)==5:
        #     div3 = tf.squeeze(div3,[1])
        
        mean, logvar = self.encoder.encoder_model4(x_hat)
        # mean1, logvar1 = self.encoder.encoder_model2(div1)
        # mean2, logvar2 = self.encoder.encoder_model3(div2)
        # mean3, logvar3 = self.encoder.encoder_model4(div3)
        print(mean.shape)
        z = self.reparameterize(mean, logvar)
        # z1 = self.reparameterize(mean1, logvar1)
        # z2 = self.reparameterize(mean2, logvar2)
        # z3 = self.reparameterize(mean3, logvar3)

        # mean_f = tf.concat([mean,mean1,mean2,mean3],1)
        print(self.en1_mean)
        x_logit = self.decoder.decoder_model4([self.en1_mean,self.en2_mean,self.en3_mean,self.en4_mean])
        # x_logit1 = self.decoder.decoder_model1([mean,mean1,mean2,mean3])
        # x_logit2 = self.decoder.decoder_model1([mean,mean1,mean2,mean3])
        # x_logit3 = self.decoder.decoder_model1([mean,mean1,mean2,mean3])

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x_hat)
        # cross_ent1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit1, labels=div_hat1)
        # cross_ent2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit2, labels=div_hat2)
        # cross_ent3 = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit3, labels=div_hat3)
       
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        # logpx_z1 = -tf.reduce_sum(cross_ent1, axis=[1, 2, 3])
        # logpx_z2 = -tf.reduce_sum(cross_ent2, axis=[1, 2, 3])
        # logpx_z3 = -tf.reduce_sum(cross_ent3, axis=[1, 2, 3])
       
        logpz = self._log_normal_pdf(z, 0., 0.)
        # logpz1 = self._log_normal_pdf(z1, 0., 0.)
        # logpz2 = self._log_normal_pdf(z2, 0., 0.)
        # logpz3 = self._log_normal_pdf(z3, 0., 0.)
       
        logqz_x = self._log_normal_pdf(z, mean, logvar)
        # logqz_x1 = self._log_normal_pdf(z1, mean1, logvar1)
        # logqz_x2 = self._log_normal_pdf(z2, mean2, logvar2)
        # logqz_x3 = self._log_normal_pdf(z3, mean3, logvar3)
       
        result = -tf.reduce_mean(logpx_z + logpz - logqz_x)
        return result
    
    def sample(self, sample_num=1,eps=None):
        if eps is None:
            #eps = tf.random.normal(shape=(sample_num, self.latent_dim))
            eps = tf.random.normal(shape=(sample_num, self.latent_dim))
        return self.decoder.decoder_model1(eps) #, apply_sigmoid=True, training=False) # ,self.decoder1(eps, apply_sigmoid=True, training=False),self.decoder2(eps, apply_sigmoid=True, training=False),self.decoder3(eps, apply_sigmoid=True, training=False)


    @tf.function
    def compute_apply_gradients(self, x, optimizer):
        print('here????')
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x,x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

# class Autoencoder(tf.keras.Model):
#     def __init__(self,*args,**kwargs):
#         super(Autoencoder, self).__init__()
#         self.latent_dim = kwargs.get('latent_dim',32)
#         kwargs['num_deconv_blocks'] = kwargs.get('num_conv_blocks',5)
#         kwargs['num_defc_blocks'] = kwargs.get('num_fc_blocks',2)
#         self.isfushion = kwargs.get('isfushion') #Raise Error if not defined
#         self.debug = kwargs.get('debug',False)
#         self._build(**kwargs)

#         self.input1 = kwargs.get('input1')
#         self.input2 = kwargs.get('input2')
#         self.input3 = kwargs.get('input3')
#         self.input4 = kwargs.get('input4')
        

#     def _build(self,**kwargs):
#         self.encoder = CNN_Encoder(**kwargs)
#         self.decoder = CNN_Decoder(**kwargs)

#         self.encoder1 = CNN_Encoder(**kwargs)
#         self.decoder1 = CNN_Decoder(**kwargs)

#         self.encoder2 = CNN_Encoder(**kwargs)
#         self.decoder2 = CNN_Decoder(**kwargs)

#         self.encoder3 = CNN_Encoder(**kwargs)
#         self.decoder3 = CNN_Decoder(**kwargs)

#     def reparameterize(self,mean,logvar):
#         #eps = tf.random.normal(shape=mean.shape)
#         eps = tf.random.normal(shape=tf.shape(mean))
#         return eps * tf.exp(logvar * .5) + mean
    
#     @tf.function
    
#     def _log_normal_pdf(self,sample, mean, logvar, raxis=1):
#         log2pi = tf.math.log(2. * np.pi)
#         return tf.reduce_sum(
#             -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
#             axis=raxis)

#     def compute_loss(self,x,x_hat):

#         div, div1, div2, div3 = tf.split(x, num_or_size_splits=4,axis=1)
#         div_hat, div_hat1, div_hat2, div_hat3 = tf.split(x_hat, num_or_size_splits=4)
#         if len(div.shape)==5:
#             div = tf.squeeze(div,[1])
#         if len(div1.shape)==5:
#             div1 = tf.squeeze(div1,[1])
#         if len(div2.shape)==5:
#             div2 = tf.squeeze(div2,[1])
#         if len(div3.shape)==5:
#             div3 = tf.squeeze(div3,[1])
        
#         mean, logvar = self.encoder(div)
#         mean1, logvar1 = self.encoder1(div1)
#         mean2, logvar2 = self.encoder2(div2)
#         mean3, logvar3 = self.encoder3(div3)

#         z = self.reparameterize(mean, logvar)
#         z1 = self.reparameterize(mean1, logvar1)
#         z2 = self.reparameterize(mean2, logvar2)
#         z3 = self.reparameterize(mean3, logvar3)

#         mean_f = tf.concat([mean,mean1,mean2,mean3],1)

#         x_logit = self.decoder(mean_f)
#         x_logit1 = self.decoder1(mean_f)
#         x_logit2 = self.decoder2(mean_f)
#         x_logit3 = self.decoder3(mean_f)

#         cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=div_hat)
#         cross_ent1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit1, labels=div_hat1)
#         cross_ent2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit2, labels=div_hat2)
#         cross_ent3 = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit3, labels=div_hat3)
       
#         logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
#         logpx_z1 = -tf.reduce_sum(cross_ent1, axis=[1, 2, 3])
#         logpx_z2 = -tf.reduce_sum(cross_ent2, axis=[1, 2, 3])
#         logpx_z3 = -tf.reduce_sum(cross_ent3, axis=[1, 2, 3])
       
#         logpz = self._log_normal_pdf(z, 0., 0.)
#         logpz1 = self._log_normal_pdf(z1, 0., 0.)
#         logpz2 = self._log_normal_pdf(z2, 0., 0.)
#         logpz3 = self._log_normal_pdf(z3, 0., 0.)
       
#         logqz_x = self._log_normal_pdf(z, mean, logvar)
#         logqz_x1 = self._log_normal_pdf(z1, mean1, logvar1)
#         logqz_x2 = self._log_normal_pdf(z2, mean2, logvar2)
#         logqz_x3 = self._log_normal_pdf(z3, mean3, logvar3)
       
#         result = -tf.reduce_mean(logpx_z+logpx_z1+logpx_z2+logpx_z3 + logpz+logpz1+logpz2+logpz3 - logqz_x-logqz_x1-logqz_x2-logqz_x3)
#         return result
    
#     @tf.function
#     def compute_apply_gradients(self, x, optimizer):
#         print('here????')
#         with tf.GradientTape() as tape:
#             loss = self.compute_loss(x,x)
#         gradients = tape.gradient(loss, self.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, self.trainable_variables))
#         return loss

#     @tf.function
#     def call(self,
#             inputs,
#             gammas=None,
#             betas=None,
#             training=True):
#         print("inputs shape in call: ",inputs.shape)
#         quit()
#         div, div1, div2, div3 = tf.split(inputs, num_or_size_splits=4, axis=1)

#         div = tf.squeeze(div,[1])
#         div1 = tf.squeeze(div1,[1])
#         div2 = tf.squeeze(div2,[1])
#         div3 = tf.squeeze(div3,[1])
#         mean, logvar = self.encoder(div,gammas,betas)
#         mean1, logvar1 = self.encoder1(div1,gammas,betas)
#         mean2, logvar2 = self.encoder2(div2,gammas,betas)
#         mean3, logvar3 = self.encoder3(div3,gammas,betas)
#         z = self.reparameterize(mean, logvar)
#         z1 = self.reparameterize(mean1, logvar1)
#         z2 = self.reparameterize(mean2, logvar2)
#         z3 = self.reparameterize(mean3, logvar3)
#         z = tf.concat([z,z1,z2,z3],1)
#         probs = self.decoder(z)
#         probs1 = self.decoder1(z)
#         probs2 = self.decoder2(z)
#         probs3 = self.decoder3(z)
#         probs = tf.concat([probs,probs1,probs2,probs3],axis=0)
#         return probs #, probs1, probs2, probs3