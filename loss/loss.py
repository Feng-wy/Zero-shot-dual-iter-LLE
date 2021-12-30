import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Conv2D, MaxPooling3D, AveragePooling3D
import numpy as np
from utils import *


def init_Guass_kernel_3x3():
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16
    kernel = tf.cast(kernel, tf.float32)
    return tf.expand_dims(tf.expand_dims(kernel, axis=[-1]), 0)


def WB_loss(x):
    r, g, b = tf.split(x, num_or_size_splits=3, axis=-1)
    mean_rgb = tf.reduce_mean(x, axis=[1, 2]) #
    mean = tf.reduce_mean(mean_rgb, axis=-1)
    m_r, m_g, m_b = tf.split(mean_rgb, num_or_size_splits=3, axis=-1)  # 沿着通道维度做切片, 得到三通道的强度均值

    r_w = r*mean/m_r
    l_r = tf.concat([r_w, tf.ones(shape=r_w.shape)], axis=-1)
    R_bal = tf.reduce_min(l_r, axis=-1)

    g_w = g * mean / m_g
    l_g = tf.concat([g_w, tf.ones(shape=g_w.shape)], axis=-1)
    G_bal = tf.reduce_min(l_g, axis=-1)

    b_w = b * mean / m_b
    l_b = tf.concat([b_w, tf.ones(shape=b_w.shape)], axis=-1)
    B_bal = tf.reduce_min(l_b, axis=-1)

    D_r_and_g = tf.reduce_mean(tf.pow(R_bal - G_bal, 2))
    D_r_and_b = tf.reduce_mean(tf.pow(R_bal - B_bal, 2))
    D_g_and_b = tf.reduce_mean(tf.pow(G_bal - B_bal, 2))
    loss_Bal = tf.pow((tf.pow(D_r_and_g, 2)) + tf.pow(D_r_and_b, 2) + tf.pow(D_g_and_b, 2), 0.5)
    return loss_Bal


def Color_loss(x):
    mean_rgb = tf.reduce_mean(x, axis=[1, 2], keepdims=True)    # 沿着长和宽维度计算均值（三通道）, 返回原tensor形状的均值张量
    m_r, m_g, m_b = tf.split(mean_rgb, num_or_size_splits=3, axis=-1)   # 沿着通道维度做切片, 得到三通道的强度均值
    D_r_and_g = tf.pow(m_r - m_g, 2)
    D_r_and_b = tf.pow(m_r - m_b, 2)
    D_g_and_b = tf.pow(m_g - m_b, 2)
    loss_color = tf.pow((tf.pow(D_r_and_g, 2)) + tf.pow(D_r_and_b, 2) + tf.pow(D_g_and_b, 2), 0.5)
    return loss_color


def TV_loss(x, weight=1.0):
    batch_size = tf.shape(x)[0].numpy()
    h_x = tf.shape(x)[1].numpy()
    w_x = tf.shape(x)[2].numpy()
    count_h = (h_x - 1) * w_x
    count_w = h_x * (w_x - 1)
    h_tv = tf.reduce_sum(tf.pow((x[:, 1:, :, :] - x[:, :h_x-1, :, :]), 2))
    w_tv = tf.reduce_sum(tf.pow((x[:, :, 1:, :] - x[:, :, :w_x-1, :]), 2))
    t_loss = 2*weight*(h_tv/count_h + w_tv/count_w)/batch_size
    return t_loss


def Bright_Contrast_enhance_Loss(org_image, enhanced_image):
    org_mean = tf.reduce_mean(org_image, axis=3, keepdims=True)
    enhanced_mean = tf.reduce_mean(enhanced_image, axis=3, keepdims=True)

    org_pool = AveragePooling2D(pool_size=(4, 4))(org_mean)
    enhanced_pool_16 = AveragePooling2D(pool_size=16, strides=16)(enhanced_mean)
    enhanced_pool = AveragePooling2D(pool_size=(4, 4), strides=4)(enhanced_mean)
    Ex = tf.reduce_mean(tf.sign(enhanced_pool - 0.5))*(enhanced_pool - 2.5*org_pool)

    D_org_left = Conv2D(kernel_initializer=init_kernel_left, kernel_size=(3, 3), padding='same', filters=1)(org_pool)
    D_org_right = Conv2D(kernel_initializer=init_kernel_right, kernel_size=(3, 3), padding='same', filters=1)(org_pool)
    D_org_up = Conv2D(kernel_initializer=init_kernel_up, kernel_size=(3, 3), padding='same', filters=1)(org_pool)
    D_org_down = Conv2D(kernel_initializer=init_kernel_down, kernel_size=(3, 3), padding='same', filters=1)(org_pool)
    D_org_L_up = Conv2D(kernel_initializer=init_kernel_L_up, kernel_size=(3, 3), padding='same', filters=1)(org_pool)
    D_org_R_up = Conv2D(kernel_initializer=init_kernel_R_up, kernel_size=(3, 3), padding='same', filters=1)(org_pool)
    D_org_L_down = Conv2D(kernel_initializer=init_kernel_L_down, kernel_size=(3, 3), padding='same', filters=1)(org_pool)
    D_org_R_down = Conv2D(kernel_initializer=init_kernel_R_down, kernel_size=(3, 3), padding='same', filters=1)(org_pool)

    D_enhanced_left = Conv2D(kernel_initializer=init_kernel_left, kernel_size=(3, 3), padding='same', filters=1)(enhanced_pool)
    D_enhaced_right = Conv2D(kernel_initializer=init_kernel_right, kernel_size=(3, 3), padding='same', filters=1)(enhanced_pool)
    D_enhanced_up = Conv2D(kernel_initializer=init_kernel_up, kernel_size=(3, 3), padding='same', filters=1)(enhanced_pool)
    D_enhanced_down = Conv2D(kernel_initializer=init_kernel_down, kernel_size=(3, 3), padding='same', filters=1)(enhanced_pool)
    D_enhanced_L_up = Conv2D(kernel_initializer=init_kernel_L_up, kernel_size=(3, 3), padding='same', filters=1)(enhanced_pool)
    D_enhanced_R_up = Conv2D(kernel_initializer=init_kernel_R_up, kernel_size=(3, 3), padding='same', filters=1)(enhanced_pool)
    D_enhanced_L_down = Conv2D(kernel_initializer=init_kernel_L_down, kernel_size=(3, 3), padding='same', filters=1)(enhanced_pool)
    D_enhanced_R_down = Conv2D(kernel_initializer=init_kernel_R_down, kernel_size=(3, 3), padding='same', filters=1)(enhanced_pool)

    D_left = tf.pow(D_org_left - D_enhanced_left, 2)
    D_right = tf.pow(D_org_right - D_enhaced_right, 2)
    D_up = tf.pow(D_org_up - D_enhanced_up, 2)
    D_down = tf.pow(D_org_down - D_enhanced_down, 2)
    D_L_up = tf.pow(D_org_L_up - D_enhanced_L_up, 2)
    D_R_up = tf.pow(D_org_R_up - D_enhanced_R_up, 2)
    D_L_down = tf.pow(D_org_L_down - D_enhanced_L_down, 2)
    D_R_down = tf.pow(D_org_R_down - D_enhanced_R_down, 2)

    loss_spa = 3*Ex + (D_left + D_right + D_up + D_down + D_L_up + D_R_up + D_L_down + D_R_down)# end2end w=3
    return loss_spa


def Bright_Channel_loss(img, E, window_size):
    window_size = window_size + np.random.randint(0, 10, dtype=np.int8)*4
    bright_channel_f = MaxPooling3D(pool_size=(3, window_size, window_size), strides=1, padding='same')(1 - img[None, :, :, :])

    '''
    pool_size: 后两位根据需要更改，当被处理图像中有大面积强度高的区域（如天空，大片亮灯）时，应加大后两位的尺寸，默认可设置为（3， 25， 25）
    '''
    bright_channel_sq_f = tf.squeeze(bright_channel_f, axis=0)
    bc_f_blur = tf.nn.conv2d(bright_channel_sq_f, filters=init_Guass_kernel_3x3(), strides=(1, 1, 1, 1), padding="SAME")
    #loss_Bc = tf.reduce_mean(tf.math.squared_difference(bright_channel_sq_f, E))
    loss_Bc = tf.norm(bc_f_blur - E, ord=1)
    # ones=tf.ones(E.shape)
    # loss_Bc = tf.norm((ones - E), ord=1)
    # print(img.shape, bright_channel_sq_in.shape, E.shape)
    return loss_Bc


def Smooth_loss_iter02(out1, out2, a0, a1, a2):
    lt0_tv = TV_loss(a0)
    lt1_tv = TV_loss(a1)
    lt2_tv = TV_loss(a2)
    out1_tv = TV_loss(out1, weight=0.01)
    out2_tv = TV_loss(out2, weight=0.01)
    Sm_loss = 0.5*lt0_tv + 0.4*lt1_tv + 0.1*lt2_tv + out1_tv + out2_tv
    return Sm_loss


def Smooth_loss_iter00(a0):
    lt0_tv = TV_loss(a0)
    #out1_tv = TV_loss(out1, weight=0.01)
    Sm_loss = 0.5*lt0_tv
    return Sm_loss


def Smooth_loss_iter01(out1, a0, a1):
    lt0_tv = TV_loss(a0)
    lt1_tv = TV_loss(a1)
    out1_tv = TV_loss(out1, weight=0.01)
    Sm_loss = 0.5*lt0_tv + 0.4*lt1_tv + out1_tv
    return Sm_loss


def Smooth_loss_iter03(out1, out2, out3, a0_1, a1_1, a2_1, a3_1):
    lt0_tv_1 = TV_loss(a0_1)
    lt0_tv_2 = TV_loss(a1_1)
    lt1_tv_1 = TV_loss(a2_1)
    lt2_tv_1 = TV_loss(a3_1)
    out1_tv = TV_loss(out1, weight=0.01)
    out2_tv = TV_loss(out2, weight=0.01)
    out3_tv = TV_loss(out3, weight=0.01)
    Sm_loss = 0.5*(lt0_tv_1+lt0_tv_2) + 0.4*lt1_tv_1 + 0.3*lt2_tv_1 + out1_tv + out2_tv + out3_tv
    return Sm_loss


def Smooth_loss_iter04(out1, out2, out3, out4, a0_1, a0_2, a1_1, a1_2, a2):
    lt0_tv_1 = TV_loss(a0_1)
    lt0_tv_2 = TV_loss(a0_2)
    lt1_tv_1 = TV_loss(a1_1)
    lt1_tv_2 = TV_loss(a1_2)
    lt2_tv = TV_loss(a2)
    out1_tv = TV_loss(out1, weight=0.01)
    out2_tv = TV_loss(out2, weight=0.01)
    out3_tv = TV_loss(out3, weight=0.01)
    out4_tv = TV_loss(out4, weight=0.01)
    Sm_loss = 0.5*(lt0_tv_1 + lt0_tv_2) + 0.4*(lt1_tv_1 + lt1_tv_2) + 0.3*lt2_tv + out1_tv + out2_tv + out3_tv + out4_tv
    return Sm_loss


def Smooth_loss_iter05(out1, out2, out3,out4,out5,  a0_1, a0_2, a1_1, a1_2, a2_1, a2_2):
    lt0_tv_1 = TV_loss(a0_1)
    lt0_tv_2 = TV_loss(a0_2)
    lt1_tv_1 = TV_loss(a1_1)
    lt1_tv_2 = TV_loss(a1_2)
    lt2_tv_1 = TV_loss(a2_1)
    lt2_tv_2 = TV_loss(a2_2)
    out1_tv = TV_loss(out1, weight=0.01)
    out2_tv = TV_loss(out2, weight=0.01)
    out3_tv = TV_loss(out3, weight=0.01)
    out4_tv = TV_loss(out4, weight=0.01)
    out5_tv = TV_loss(out5, weight=0.01)
    Sm_loss = 0.5*(lt0_tv_1 + lt0_tv_2) + 0.4*(lt1_tv_1 + lt1_tv_2) + 0.3*(lt2_tv_1 + lt2_tv_2) + out1_tv + out2_tv + out3_tv + out4_tv + out5_tv
    return Sm_loss


def Smooth_loss_iter06(out1, out2, out3, out4, out5, out6,  a0_1, a0_2, a0_3, a1_1, a1_2, a2_1, a2_2):
    lt0_tv_1 = TV_loss(a0_1)
    lt0_tv_2 = TV_loss(a0_2)
    lt0_tv_3 = TV_loss(a0_3)
    lt1_tv_1 = TV_loss(a1_1)
    lt1_tv_2 = TV_loss(a1_2)
    lt2_tv_1 = TV_loss(a2_1)
    lt2_tv_2 = TV_loss(a2_2)
    out1_tv = TV_loss(out1, weight=0.01)
    out2_tv = TV_loss(out2, weight=0.01)
    out3_tv = TV_loss(out3, weight=0.01)
    out4_tv = TV_loss(out4, weight=0.01)
    out5_tv = TV_loss(out5, weight=0.01)
    out6_tv = TV_loss(out6, weight=0.01)
    Sm_loss = 0.5*(lt0_tv_1 + lt0_tv_2 + lt0_tv_3) + 0.4*(lt1_tv_1 + lt1_tv_2) + 0.3*(lt2_tv_1 + lt2_tv_2) + out1_tv + out2_tv + out3_tv + out4_tv + out5_tv + out6_tv
    return Sm_loss