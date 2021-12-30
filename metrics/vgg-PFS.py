import tensorflow as tf
import glob
import numpy as np
import os
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def preprocess(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    # image = tf.image.resize(image, size=(400, 600),
    #                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # image = tf.image.random_crop(image, size=(self.opt.crop_size, self.opt.crop_size))
    # image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0
    return image


img_path_my = glob.glob('C:/Users/wyf1998/PycharmProjects/myLowLightenhance/result_320/*.png')
# img_path_my.sort(key=lambda x: int(x.split('test_')[1].split('.png')[0]))
# img_path_my.sort(key=lambda x: int(x.split('\\')[1].split('.png')[0]))
print(len(img_path_my))
gt_path_pre = glob.glob('C:/Users/wyf1998/PycharmProjects/myLowLightenhance/data/Normal/*.png')


def VGG_loss(pred, reference):
    conv_base = tf.keras.applications.VGG16(weights='imagenet', input_shape=(400, 600, 3), include_top=False)
    # 获取网络中间层的输出
    layer_names = ['block5_conv3', 'block4_conv3', 'block3_conv3']
    layers_output = [conv_base.get_layer(layer_name).output for layer_name in layer_names]

    multi_out_model = tf.keras.models.Model(inputs=conv_base.input, outputs=layers_output)
    multi_out_model.trainable = False
    out_1_pre, out_2_pre, out_3_pre = multi_out_model(pred)
    out_1_ref, out_2_ref, out_3_ref = multi_out_model(reference)
    mean_absolute_errors = tf.reduce_mean(tf.losses.mean_absolute_error(out_1_pre, out_1_ref)) + \
        tf.reduce_mean(tf.losses.mean_absolute_error(out_2_pre, out_2_ref)) + \
        tf.reduce_mean(tf.losses.mean_absolute_error(out_3_pre, out_3_ref))

    return mean_absolute_errors


mae_vgg_list = []
for i in range(len(img_path_my)):
    img_pre = preprocess(img_path_my[i])
    img_pre_batch = tf.expand_dims(img_pre, axis=0)
    img_ref = preprocess(gt_path_pre[i])
    img_ref_batch = tf.expand_dims(img_ref, axis=0)
    mae_vgg = VGG_loss(img_pre_batch, img_ref_batch)
    print('mae_vgg:', mae_vgg.numpy())
    mae_vgg_list.append(mae_vgg.numpy())

print(mae_vgg_list)
total_mae_vgg = np.mean(mae_vgg_list)
length = len(mae_vgg_list)
print('total_nums:{},total_mae_vgg:{}'.format(length, total_mae_vgg))