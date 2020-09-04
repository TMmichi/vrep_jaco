block_setting = dict()

#############################################################
#####################   input definition   ##################
#############################################################
inputs = dict()
inputs['width'] = 640
inputs['height'] = 480
inputs['latent_size'] = 64
inputs['name'] = 'inputs'



#############################################################
#####################   conv blk setting   ##################
#############################################################
#Conv block1 settings
conv_blk1 = dict()
conv_blk1['with_batch_norm'] = False
conv_blk1['n_features'] = 16
conv_blk1['kernel_size'] = 8
conv_blk1['stride'] = (2,2)
conv_blk1['padding'] = 'SAME' # conv2d same -> if stride == 1 then the image size is consistent
conv_blk1['activation']='relu'
conv_blk1['dropout']=True
conv_blk1['name'] = 'conv_blk1'
conv_blk1['width'] = inputs['width']/conv_blk1['stride'][0]
conv_blk1['height'] = inputs['height']/conv_blk1['stride'][1]

#Conv block2 settings
conv_blk2 = dict()
conv_blk2['with_batch_norm'] = False
conv_blk2['n_features'] = 32
conv_blk2['kernel_size'] = 4
conv_blk2['stride'] = (2,2)
conv_blk2['padding'] = 'SAME'
conv_blk2['activation']='relu'
conv_blk2['dropout']=True
conv_blk2['name'] = 'conv_blk2'
conv_blk2['width'] = conv_blk1['width']/conv_blk2['stride'][0]
conv_blk2['height'] = conv_blk1['height']/conv_blk2['stride'][1]

#Conv block3 settings
conv_blk3 = dict()
conv_blk3['with_batch_norm'] = False
conv_blk3['n_features'] = 32
conv_blk3['kernel_size'] = 4
conv_blk3['stride'] = (2,2)
conv_blk3['padding'] = 'SAME'
conv_blk3['activation']='relu'
conv_blk3['dropout']=True
conv_blk3['name'] = 'conv_blk3'
conv_blk3['width'] = conv_blk2['width']/conv_blk3['stride'][0]
conv_blk3['height'] = conv_blk2['height']/conv_blk3['stride'][1]

#Conv block4 settings
conv_blk4 = dict()
conv_blk4['with_batch_norm'] = False
conv_blk4['n_features'] = 64
conv_blk4['kernel_size'] = 3
conv_blk4['stride'] = (2,2)
conv_blk4['padding'] = 'SAME'
conv_blk4['activation']='relu'
conv_blk4['dropout']=True
conv_blk4['name'] = 'conv_blk4'
conv_blk4['width'] = conv_blk3['width']/conv_blk4['stride'][0]
conv_blk4['height'] = conv_blk3['height']/conv_blk4['stride'][1]

#Conv block final settings
conv_blk_final = dict()
conv_blk_final['with_batch_norm'] = False
conv_blk_final['n_features'] = 128
conv_blk_final['kernel_size'] = 3
conv_blk_final['stride'] = (2,2)
conv_blk_final['padding'] = 'SAME'
conv_blk_final['activation']='relu'
conv_blk_final['dropout']=True
conv_blk_final['name'] = 'conv_blk_final'
conv_blk_final['width'] = conv_blk4['width']/conv_blk3['stride'][0]
conv_blk_final['height'] = conv_blk4['height']/conv_blk3['stride'][1]



#############################################################
################   fc blk before merge setting   ############
#############################################################
#FC bm layer1 settings
conv_fc_bm_1 = dict()
conv_fc_bm_1['units'] = 512
conv_fc_bm_1['survival_prob'] = 0.6
conv_fc_bm_1['dropout'] = True
conv_fc_bm_1['name'] = 'conv_fc_bm_1'

#FC bm layer2 settings
conv_fc_bm_2 = dict()
conv_fc_bm_2['units'] = 2*inputs['latent_size'] # latent vector dim
conv_fc_bm_2['survival_prob'] = 0.6
conv_fc_bm_2['dropout'] = False
conv_fc_bm_2['name'] = 'conv_fc_bm_2'



#############################################################
################   fc blk after merge setting   #############
#############################################################
#FC am layer1 settings
conv_fc_am_1 = dict()
conv_fc_am_1['units'] = 512
conv_fc_am_1['survival_prob'] = 0.6
conv_fc_am_1['dropout'] = True
conv_fc_am_1['name'] = 'conv_fc_am_1'

#FC am layer2 settings
conv_fc_am_2 = dict()
conv_fc_am_2['units'] = 512
conv_fc_am_2['survival_prob'] = 0.6
conv_fc_am_2['dropout'] = False
conv_fc_am_2['name'] = 'conv_fc_am_2'



#############################################################
###############   fc blk before divide setting   ############
#############################################################
#FC bd layer1 settings
deconv_fc_bd_1 = dict()
deconv_fc_bd_1['units'] = 512
deconv_fc_bd_1['survival_prob'] = 0.6
deconv_fc_bd_1['dropout'] = True
deconv_fc_bd_1['name'] = 'deconv_fc_bd_1'

#FC bd layer2 settings
deconv_fc_bd_2 = dict()
deconv_fc_bd_2['units'] = 512
deconv_fc_bd_2['survival_prob'] = 0.6
deconv_fc_bd_2['dropout'] = False
deconv_fc_bd_2['name'] = 'deconv_fc_bd_2'



#############################################################
###############   fc blk after divide setting   #############
#############################################################
#FC ad layer1 settings
deconv_fc_ad_1 = dict()
deconv_fc_ad_1['units'] = 512
deconv_fc_ad_1['survival_prob'] = 0.6
deconv_fc_ad_1['dropout'] = True
deconv_fc_ad_1['name'] = 'deconv_fc_ad_1'

#FC ad layer2 settings
deconv_fc_ad_2 = dict()
deconv_fc_ad_2['units'] = conv_blk_final['height']*conv_blk_final['width']*conv_blk_final['n_features']
deconv_fc_ad_2['survival_prob'] = 0.6
deconv_fc_ad_2['dropout'] = True
deconv_fc_ad_2['name'] = 'deconv_fc_ad_2'



#############################################################
####################   deconv blk setting   #################
#############################################################
#DeConv block1 settings
deconv_blk1 = dict()
deconv_blk1['with_batch_norm'] = False
deconv_blk1['n_features'] = 64
deconv_blk1['kernel_size'] = 3
deconv_blk1['stride'] = (2,2)
deconv_blk1['padding'] = 'SAME'
deconv_blk1['activation']='relu'
deconv_blk1['dropout'] = True
deconv_blk1['name'] = 'deconv_blk1'
deconv_blk1['width'] = conv_blk_final['width']*deconv_blk1['stride'][0]
deconv_blk1['height'] = conv_blk_final['height']*deconv_blk1['stride'][1]


#DeConv block2 settings
deconv_blk2 = dict()
deconv_blk2['with_batch_norm'] = False
deconv_blk2['n_features'] = 32
deconv_blk2['kernel_size'] = 3
deconv_blk2['stride'] = (2,2)
deconv_blk2['padding'] = 'SAME'
deconv_blk2['activation']='relu'
deconv_blk2['dropout'] = True
deconv_blk2['name'] = 'deconv_blk2'
deconv_blk2['width'] = deconv_blk1['width']*deconv_blk2['stride'][0]
deconv_blk2['height'] = deconv_blk1['height']*deconv_blk2['stride'][1]

#DeConv block3 settings
deconv_blk3 = dict()
deconv_blk3['with_batch_norm'] = False
deconv_blk3['n_features'] = 32
deconv_blk3['kernel_size'] = 4
deconv_blk3['stride'] = (2,2)
deconv_blk3['padding'] = 'SAME'
deconv_blk3['activation']='relu'
deconv_blk3['dropout'] = True
deconv_blk3['name'] = 'deconv_blk3'
deconv_blk3['width'] = deconv_blk2['width']*deconv_blk3['stride'][0]
deconv_blk3['height'] = deconv_blk2['height']*deconv_blk3['stride'][1]

#DeConv block4 settings
deconv_blk4 = dict()
deconv_blk4['with_batch_norm'] = False
deconv_blk4['n_features'] = 16
deconv_blk4['kernel_size'] = 4
deconv_blk4['stride'] = (2,2)
deconv_blk4['padding'] = 'SAME'
deconv_blk4['activation']='relu'
deconv_blk4['dropout'] = True
deconv_blk4['name'] = 'deconv_blk4'
deconv_blk4['width'] = deconv_blk3['width']*deconv_blk4['stride'][0]
deconv_blk4['height'] = deconv_blk3['height']*deconv_blk4['stride'][1]

#DeConv block5 settings
deconv_blk5 = dict()
deconv_blk5['with_batch_norm'] = False
deconv_blk5['n_features'] = 1
deconv_blk5['kernel_size'] = 8
deconv_blk5['stride'] = (2,2)
deconv_blk5['padding'] = 'SAME'
deconv_blk5['activation']='relu'
deconv_blk5['dropout'] = True
deconv_blk5['name'] = 'deconv_blk5'
deconv_blk5['width'] = deconv_blk4['width']*deconv_blk5['stride'][0]
deconv_blk5['height'] = deconv_blk4['height']*deconv_blk5['stride'][1]



#############################################################
###################   build setting config   ################
#############################################################
#Setting Stack
block_setting['conv_block1'] = conv_blk1
block_setting['conv_block2'] = conv_blk2
block_setting['conv_block3'] = conv_blk3
block_setting['conv_block4'] = conv_blk4
block_setting['conv_block_final'] = conv_blk_final

block_setting['conv_fc_bm_1'] = conv_fc_bm_1
block_setting['conv_fc_bm_2'] = conv_fc_bm_2
block_setting['conv_fc_am_1'] = conv_fc_am_1
block_setting['conv_fc_am_2'] = conv_fc_am_2

block_setting['deconv_fc_bd_1'] = deconv_fc_bd_1
block_setting['deconv_fc_bd_2'] = deconv_fc_bd_2
block_setting['deconv_fc_ad_1'] = deconv_fc_ad_1
block_setting['deconv_fc_ad_2'] = deconv_fc_ad_2

block_setting['deconv_block1'] = deconv_blk1
block_setting['deconv_block2'] = deconv_blk2
block_setting['deconv_block3'] = deconv_blk3
block_setting['deconv_block4'] = deconv_blk4
block_setting['deconv_block5'] = deconv_blk5



if __name__ == "__main__":
    for key, value in block_setting.items():
        print(key)
        print(value)
