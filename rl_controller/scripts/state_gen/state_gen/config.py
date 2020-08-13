block_setting = dict()

input_size = dict()
input_size['width'] = 640
input_size['height'] = 480

info = dict()
info['en_conv_num'] = 5
info['en_fc_num'] = 2
info['de_conv_num'] = 6
info['de_fc_num'] = 2

#Conv block1 settings
conv_blk1 = dict()
conv_blk1['type'] = 'conv'
conv_blk1['with_batch_norm'] = False
conv_blk1['n_features'] = 16
conv_blk1['kernel_size'] = 8
conv_blk1['stride'] = (2,2)
conv_blk1['padding'] = 'SAME' # or valid
conv_blk1['activation']='relu'
if conv_blk1['padding']=='SAME':
    conv_blk1['output_w'] = input_size['width']
else:
    conv_blk1['output_w'] = (input_size['width'] - conv_blk1['kernel_size'])/conv_blk1['stride'][0]+1
if conv_blk1['padding']=='SAME':
    conv_blk1['output_h'] = input_size['height']
else:
    conv_blk1['output_h'] = (input_size['height'] - conv_blk1['kernel_size'])/conv_blk1['stride'][1]+1


#Conv block2 settings
conv_blk2 = dict()
conv_blk2['type'] = 'conv'
conv_blk2['with_batch_norm'] = False
conv_blk2['n_features'] = 32
conv_blk2['kernel_size'] = 4
conv_blk2['stride'] = (2,2)
conv_blk2['padding'] = 'SAME'
conv_blk2['activation']='relu'
if conv_blk2['padding']=='SAME':
    conv_blk2['output_w'] = conv_blk1['output_w']
else:
    conv_blk2['output_w'] = (conv_blk1['output_w'] - conv_blk2['kernel_size'])/conv_blk2['stride'][0]+1
if conv_blk2['padding']=='SAME':
    conv_blk2['output_h'] = conv_blk1['output_h']
else:
    conv_blk2['output_h'] = (conv_blk1['output_h'] - conv_blk2['kernel_size'])/conv_blk2['stride'][1]+1


#Conv block3 settings
conv_blk3 = dict()
conv_blk3['type'] = 'conv'
conv_blk3['with_batch_norm'] = False
conv_blk3['n_features'] = 32
conv_blk3['kernel_size'] = 4
conv_blk3['stride'] = (2,2)
conv_blk3['padding'] = 'SAME'
conv_blk3['activation']='relu'
if conv_blk3['padding']=='SAME':
    conv_blk3['output_w'] = conv_blk2['output_w']
else:
    conv_blk3['output_w'] = (conv_blk2['output_w'] - conv_blk3['kernel_size'])/conv_blk3['stride'][0]+1
if conv_blk3['padding']=='SAME':
    conv_blk3['output_h'] = conv_blk2['output_h']
else:
    conv_blk3['output_h'] = (conv_blk2['output_h'] - conv_blk3['kernel_size'])/conv_blk3['stride'][1]+1


#Conv block4 settings
conv_blk4 = dict()
conv_blk4['type'] = 'conv'
conv_blk4['with_batch_norm'] = False
conv_blk4['n_features'] = 64
conv_blk4['kernel_size'] = 3
conv_blk4['stride'] = (2,2)
conv_blk4['padding'] = 'SAME'
conv_blk4['activation']='relu'
if conv_blk4['padding']=='SAME':
    conv_blk4['output_w'] = conv_blk3['output_w']
else:
    conv_blk4['output_w'] = (conv_blk3['output_w'] - conv_blk4['kernel_size'])/conv_blk4['stride'][0]+1
if conv_blk4['padding']=='SAME':
    conv_blk4['output_h'] = conv_blk3['output_h']
else:
    conv_blk4['output_h'] = (conv_blk3['output_h'] - conv_blk4['kernel_size'])/conv_blk4['stride'][1]+1


#Conv block5 settings
conv_blkf = dict()
conv_blkf['type'] = 'conv'
conv_blkf['with_batch_norm'] = False
conv_blkf['n_features'] = 128
conv_blkf['kernel_size'] = 3
conv_blkf['stride'] = (2,2)
conv_blkf['padding'] = 'SAME'
conv_blkf['activation']='relu'
if conv_blkf['padding']=='SAME':
    conv_blkf['output_w'] = conv_blk4['output_w']
else:
    conv_blkf['output_w'] = (conv_blk4['output_w'] - conv_blkf['kernel_size'])/conv_blkf['stride'][0]+1
if conv_blkf['padding']=='SAME':
    conv_blkf['output_h'] = conv_blk4['output_h']
else:
    conv_blkf['output_h'] = (conv_blk4['output_h'] - conv_blkf['kernel_size'])/conv_blkf['stride'][1]+1


#FC layer1 settings
fc_1 = dict()
fc_1['type'] = 'fc'
fc_1['units'] = 512
fc_1['survival_prob'] = 0.6

#FC layer2 settings
fc_2 = dict()
fc_2['type'] = 'fc'
fc_2['units'] = 512
fc_2['survival_prob'] = 0.6

#Latent layer settings
latent_layer = dict()
latent_layer['survival_prob'] = 0.6

#DeFC layer1 settings
defc_1 = dict()
defc_1['type'] = 'defc'
defc_1['units'] = 512
defc_1['survival_prob'] = 0.6

#DeFC layer2 settings
defc_2 = dict()
defc_2['type'] = 'defc'
defc_2['units'] = 512
defc_2['survival_prob'] = 0.6

#Reshape layer settings
reshape = dict()
reshape['width'] = conv_blkf['output_w']
reshape['height'] = conv_blkf['output_h']
reshape['n_features'] = conv_blkf['n_features']
reshape['units'] = reshape['width']*reshape['height']*reshape['n_features']
reshape['survival_prob'] = 0.6

#DeConv block1 settings
deconv_blk1 = dict()
deconv_blk1['with_batch_norm'] = False
deconv_blk1['n_features'] = 64
deconv_blk1['kernel_size'] = 3
deconv_blk1['stride'] = (2,2)
deconv_blk1['padding'] = 'SAME'
deconv_blk1['activation']='relu'
if deconv_blk1['padding']=='SAME':
    deconv_blk1['output_w'] = reshape['width']
else:
    deconv_blk1['output_w'] = (reshape['width'] - deconv_blk1['kernel_size'])/deconv_blk1['stride'][0]+1
if deconv_blk1['padding']=='SAME':
    deconv_blk1['output_h'] = reshape['height']
else:
    deconv_blk1['output_h'] = (reshape['height'] - deconv_blk1['kernel_size'])/deconv_blk1['stride'][1]+1

#DeConv block2 settings
deconv_blk2 = dict()
deconv_blk2['with_batch_norm'] = False
deconv_blk2['n_features'] = 32
deconv_blk2['kernel_size'] = 3
deconv_blk2['stride'] = (2,2)
deconv_blk2['padding'] = 'SAME'
deconv_blk2['activation']='relu'
if deconv_blk2['padding']=='SAME':
    deconv_blk2['output_w'] = deconv_blk1['output_w']
else:
    deconv_blk2['output_w'] = (deconv_blk1['output_w'] - deconv_blk2['kernel_size'])/deconv_blk2['stride'][0]+1
if deconv_blk2['padding']=='SAME':
    deconv_blk2['output_h'] = deconv_blk1['output_h']
else:
    deconv_blk2['output_h'] = (deconv_blk1['output_h'] - deconv_blk2['kernel_size'])/deconv_blk2['stride'][1]+1

#DeConv block3 settings
deconv_blk3 = dict()
deconv_blk3['with_batch_norm'] = False
deconv_blk3['n_features'] = 32
deconv_blk3['kernel_size'] = 4
deconv_blk3['stride'] = (2,2)
deconv_blk3['padding'] = 'SAME'
deconv_blk3['activation']='relu'
if deconv_blk3['padding']=='SAME':
    deconv_blk3['output_w'] = deconv_blk2['output_w']
else:
    deconv_blk3['output_w'] = (deconv_blk2['output_w'] - deconv_blk3['kernel_size'])/deconv_blk3['stride'][0]+1
if deconv_blk3['padding']=='SAME':
    deconv_blk3['output_h'] = deconv_blk2['output_h']
else:
    deconv_blk3['output_h'] = (deconv_blk2['output_h'] - deconv_blk3['kernel_size'])/deconv_blk3['stride'][1]+1

#DeConv block4 settings
deconv_blk4 = dict()
deconv_blk4['with_batch_norm'] = False
deconv_blk4['n_features'] = 16
deconv_blk4['kernel_size'] = 4
deconv_blk4['stride'] = (2,2)
deconv_blk4['padding'] = 'SAME'
deconv_blk4['activation']='relu'
if deconv_blk4['padding']=='SAME':
    deconv_blk4['output_w'] = deconv_blk3['output_w']
else:
    deconv_blk4['output_w'] = (deconv_blk3['output_w'] - deconv_blk4['kernel_size'])/deconv_blk4['stride'][0]+1
if deconv_blk4['padding']=='SAME':
    deconv_blk4['output_h'] = deconv_blk3['output_h']
else:
    deconv_blk4['output_h'] = (deconv_blk3['output_h'] - deconv_blk4['kernel_size'])/deconv_blk4['stride'][1]+1

#DeConv block5 settings
deconv_blk5 = dict()
deconv_blk5['with_batch_norm'] = False
deconv_blk5['n_features'] = 1
deconv_blk5['kernel_size'] = 8
deconv_blk5['stride'] = (2,2)
deconv_blk5['padding'] = 'SAME'
deconv_blk5['activation']='relu'
if deconv_blk5['padding']=='SAME':
    deconv_blk5['output_w'] = deconv_blk4['output_w']
else:
    deconv_blk5['output_w'] = (deconv_blk4['output_w'] - deconv_blk5['kernel_size'])/deconv_blk5['stride'][0]+1
if deconv_blk5['padding']=='SAME':
    deconv_blk5['output_h'] = deconv_blk4['output_h']
else:
    deconv_blk5['output_h'] = (deconv_blk4['output_h'] - deconv_blk5['kernel_size'])/deconv_blk5['stride'][1]+1

#DeConv output settings
deconv_blkf = dict()
deconv_blkf['with_batch_norm'] = False
deconv_blkf['n_features'] = 1
deconv_blkf['kernel_size'] = 8
deconv_blkf['stride'] = (1,1)
deconv_blkf['padding'] = 'SAME'
deconv_blkf['activation']='relu'
if deconv_blkf['padding']=='SAME':
    deconv_blkf['output_w'] = deconv_blk5['output_w']
else:
    deconv_blkf['output_w'] = (deconv_blk5['output_w'] - deconv_blkf['kernel_size'])/deconv_blkf['stride'][0]+1
if deconv_blkf['padding']=='SAME':
    deconv_blkf['output_h'] = deconv_blk5['output_h']
else:
    deconv_blkf['output_h'] = (deconv_blk5['output_h'] - deconv_blkf['kernel_size'])/deconv_blkf['stride'][1]+1

#Setting Stack
block_setting['conv_block1'] = conv_blk1
block_setting['conv_block2'] = conv_blk2
block_setting['conv_block3'] = conv_blk3
block_setting['conv_block4'] = conv_blk4
block_setting['conv_blockf'] = conv_blkf

block_setting['fc_block1'] = fc_1
block_setting['fc_block2'] = fc_2

block_setting['latent_layer'] = latent_layer

block_setting['defc_block1'] = defc_1
block_setting['defc_block2'] = defc_2

block_setting['reshape'] = reshape

block_setting['deconv_block1'] = deconv_blk1
block_setting['deconv_block2'] = deconv_blk2
block_setting['deconv_block3'] = deconv_blk3
block_setting['deconv_block4'] = deconv_blk4
block_setting['deconv_block5'] = deconv_blk5
block_setting['deconv_blkf'] = deconv_blkf
