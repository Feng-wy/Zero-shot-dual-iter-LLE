import tensorflow as tf
import numpy as np
import math
import glob
import matplotlib.pyplot as plt


def preprocess(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, size=(400, 600),
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # image = tf.image.random_crop(image, size=(self.opt.crop_size, self.opt.crop_size))
    # image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0
    return image


img_path_my = glob.glob('./result_lol/*.png')
print(img_path_my)
# img_path_my = np.sort(img_path_my)
# img_path_my.sort(key=lambda x: int(x.split('\\')[1].split('.png')[0]))
print(img_path_my)
gt_path = glob.glob('./data/Normal/*.png')
gt_path_pre = glob.glob('./data/high/*.png')
gt_path = np.sort(gt_path)
# print(gt_path_pre)
psnr_list = []
ssim_list = []
mae_list = []
for i in range(len(img_path_my)):
    img_en = preprocess(img_path_my[i])
    img_gt = preprocess(gt_path[i])
    psnr = tf.image.psnr(img_en, img_gt, max_val=1.0)
    ssim = tf.image.ssim(img_en, img_gt, max_val=1.0)
    mae = tf.reduce_mean(tf.losses.mean_absolute_error(img_gt, img_en))
    psnr_list.append(psnr.numpy())
    ssim_list.append(ssim.numpy())
    mae_list.append(mae.numpy())

print('PSNR', psnr_list)
print('SSIM', ssim_list)
print('MAE', mae_list)
print(len(psnr_list), len(ssim_list), len(mae_list))
psnr_mean = np.mean(psnr_list)
ssim_mean = np.mean(ssim_list)
mae_mean = np.sum(mae_list)
print('psnr_mean:{:.2f}, ssim_mean:{:.3f}, mae_mean:{:.3f}'.format(psnr_mean, ssim_mean, mae_mean))


