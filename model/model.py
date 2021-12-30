# from tensorflow.keras import Input, Model, Sequential
# from tensorflow.keras.layers import Conv2D, Concatenate, ReLU, MaxPooling3D, Conv2DTranspose
import tensorflow as tf
import numpy as np


def init_Guass_kernel_3x3():
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16
    kernel = tf.cast(kernel, tf.float32)
    return tf.expand_dims(tf.expand_dims(kernel, axis=[-1]), 0)


class A0_Net(tf.keras.Model):
    def __init__(self):
        super(A0_Net, self).__init__()
        self.Concat = tf.keras.layers.Concatenate(axis=-1)
        self.down_0 = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, activation='relu', padding='same'),
             tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same')]
        )
        self.A0_conv1_9 = tf.keras.layers.Conv2D(32, kernel_size=(9, 9), strides=(1, 1), padding='same', activation='relu')
        self.A0_conv1_7 = tf.keras.layers.Conv2D(32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')
        self.A0_conv2_7 = tf.keras.layers.Conv2D(32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')
        self.A0_conv2_5 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.A0_conv3_3_1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.A0_conv3_3_2 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.A0_conv3_3_3 = tf.keras.layers.Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='tanh')

    def call(self, inputs, training=True, mask=None):
        x = inputs
        A0_x1 = self.down_0(x)
        A0_x2 = self.Concat([self.A0_conv1_9(A0_x1), self.A0_conv1_7(A0_x1)])
        A0_x3 = tf.add(A0_x1, A0_x2)
        A0_x4 = self.Concat([self.A0_conv2_7(A0_x3), self.A0_conv2_5(A0_x3)])
        A0_x5 = tf.add(A0_x3, A0_x4)
        A0_x6 = self.A0_conv3_3_1(A0_x5)
        A0_x6 = self.A0_conv3_3_2(A0_x6)
        A0_x6_1 = self.A0_conv3_3_3(A0_x6)
        A0_x6_2 = self.A0_conv3_3_3(A0_x6)
        A0_x6_3 = self.A0_conv3_3_3(A0_x6)
        return A0_x6_1, A0_x6_2, A0_x6_3


class A1_Net(tf.keras.Model):
    def __init__(self):
        super(A1_Net, self).__init__()
        self.Concat = tf.keras.layers.Concatenate(axis=-1)
        self.down_1 = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, activation='relu', padding='same'),
             tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, activation='relu', padding='same')]
        )  # 400x600 -> 200x300
        #   in:200x300 out:400x600
        self.A1_conv1_7 = tf.keras.layers.Conv2D(32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')
        self.A1_conv1_5 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.A1_conv2_5 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.A1_conv2_3 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.A1_conv3_3_1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.A1_conv3_3_2 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.A1_up1 = tf.keras.layers.Conv2DTranspose(3, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='tanh')

    def call(self, inputs, training=True, mask=None):
        x = inputs
        A1_x1 = self.down_1(x)
        A1_x2 = self.Concat([self.A1_conv1_7(A1_x1), self.A1_conv1_5(A1_x1)])
        A1_x3 = tf.add(A1_x1, A1_x2)
        A1_x4 = self.Concat([self.A1_conv2_5(A1_x3), self.A1_conv2_3(A1_x3)])
        A1_x5 = tf.add(A1_x3, A1_x4)
        A1_x6 = self.A1_conv3_3_1(A1_x5)
        # print(A2_x6.shape)
        A1_x6 = self.A1_conv3_3_2(A1_x6)
        A1_x6_1 = self.A1_up1(A1_x6)
        A1_x6_2 = self.A1_up1(A1_x6)
        return A1_x6_1, A1_x6_2


class A2_Net(tf.keras.Model):
    def __init__(self):
        super(A2_Net, self).__init__()
        self.Concat = tf.keras.layers.Concatenate(axis=-1)
        self.down_2 = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, activation='relu', padding='same'),
             tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, activation='relu', padding='same')]
        )  # 200x300 -> 100x150
        #   in:100x150 out:400x600
        self.A2_conv1_5 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.A2_conv1_3 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.A2_conv2_3 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.A2_conv2_1 = tf.keras.layers.Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.A2_conv3 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.A2_up1 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')
        self.A2_up2 = tf.keras.layers.Conv2DTranspose(3, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='tanh')

    def call(self, inputs, training=True, mask=None):
        x = inputs
        A2_x1 = self.down_2(x)
        A2_x2 = self.Concat([self.A2_conv1_5(A2_x1), self.A2_conv1_3(A2_x1)])
        A2_x3 = tf.add(A2_x1, A2_x2)
        A2_x4 = self.Concat([self.A2_conv2_3(A2_x3), self.A2_conv2_1(A2_x3)])
        A2_x5 = tf.add(A2_x3, A2_x4)
        A2_x6 = self.A2_conv3(A2_x5)
        # print(A3_x6.shape)
        A2_x7 = self.A2_up1(A2_x6)
        A2_x8_1 = self.A2_up2(A2_x7)
        A2_x8_2 = self.A2_up2(A2_x7)
        return A2_x8_1, A2_x8_2


class Attn_AL(tf.keras.Model):
    def __init__(self):
        super(Attn_AL, self).__init__()
        self.conv = tf.keras.Sequential([tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, activation='relu', padding='same'),
                                        tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, activation='relu', padding='same')])
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv3 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv4 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, activation='relu', padding='same')

        self.Concat= tf.keras.layers.Concatenate(axis=-1)
        self.convat_sigmoid = tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid', use_bias=False)
        self.convout_sigmoid = tf.keras.layers.Conv2D(3, kernel_size=3, strides=1, padding='same', activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        bright_channel_f = tf.keras.layers.MaxPooling3D(pool_size=(3, 4, 4), strides=1, padding='same')(1 - inputs[None, :, :, :])
        bright_channel_sq_f = tf.squeeze(bright_channel_f, axis=0)
        bc_f_blur = tf.nn.conv2d(bright_channel_sq_f, filters=init_Guass_kernel_3x3(), strides=(1, 1, 1, 1),
                                 padding="SAME")

        x = self.conv(tf.concat([bc_f_blur, inputs], axis=-1))#attn
        F_1 = self.conv2(self.conv1(x))
        F_1 = tf.multiply(F_1, bc_f_blur)#attn
        F_2 = tf.add(x, F_1)
        F_3 = self.conv4(self.conv3(F_2))#attn
        F_3 = tf.multiply(F_3, bc_f_blur)
        Features = tf.add(F_2, F_3)

        avg_out = tf.reduce_mean(Features, axis=-1, keepdims=True)
        #max_out = tf.reduce_max(Features, axis=-1, keepdims=True)  #

        #att_out = self.Concat([avg_out, max_out]) #
        att_out_sigmoid = self.convat_sigmoid(avg_out)

        E_out = self.convout_sigmoid(tf.multiply(Features, att_out_sigmoid))
        return E_out


class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.A0 = A0_Net()
        self.A1 = A1_Net()
        self.A2 = A2_Net()
        self.E = Attn_AL()

    def call(self, inputs, training=True, mask=None):
        x = inputs
        a0_1, a0_2, _ = self.A0(x)
        a1_1, _ = self.A1(x)
        a2_1, _ = self.A2(x)
        e = self.E(x)
        b = 1.0

        out1 = (a0_1 * x + b)*(x - 1) + 1
        out2 = (a0_2 * out1 + b)*(out1 - 1) + 1
        out3 = (a1_1 * out2 + b)*(out2 - 1) + 1
        out4 = (a2_1 * out3 + b)*(out3 - 1) + 1
        # out5 = (a1_2 * out4 + b)*(out4 - e) + e
        # out6 = (a2_1 * out5 + b)*(out5 - e) + e
        # out7 = (a2_2 * out6 + b)*(out6 - e) + e
        return out1, out2, out3, out4, a0_1, a0_2, a1_1, a2_1, e


model = Net()
model.build(input_shape=(None, 400, 600, 3))
model.summary()




