import tensorflow as tf
from utils import *
from model.model import model
from loss.loss import *
import os


img_path = glob.glob('./data/Forza4/*.png')#图片路径
img = preprocess(img_path[1])#选择对应图片的索引
img_batch = tf.expand_dims(img, axis=0)

train_loss = tf.keras.metrics.Mean('train_loss')
test_loss = tf.keras.metrics.Mean('test_loss')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
result_dir = './Forza4_result'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


def pipeline_step(images):
    with tf.GradientTape() as tape:
        out1, out2, out3, out4, a0_1, a0_2, a1_1, a2_1, e = model(images)
        loss_BCe = Bright_Contrast_enhance_Loss(images, out4)
        loss_col = 5*Color_loss(out4)
        # #loss_Wb = 2.0*WB_loss(out4)
        loss_Bc = 0.00001*Bright_Channel_loss(images, e, window_size=4)
        loss_sm = 150*Smooth_loss_iter03(out1, out2, out3, a0_1, a0_2, a1_1, a2_1)

        total_loss = loss_BCe + loss_col + loss_Bc + loss_sm
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(total_loss)
    return out4, e


def train(img_batch,  iter_nums):
    img_in = img_batch
    for i in range(1, iter_nums+1):
        out4,  E = pipeline_step(img_in)
        print('iter{}, loss={}'.format(i, train_loss.result()))
        img_pred = tf.squeeze(out4)
        img_png = img_pred*255.0
        img_png = tf.cast(img_png, tf.uint8)
        img_png = tf.image.encode_png(img_png)
        file_name = result_dir + '/' + str(i) + '.png'
        with tf.io.gfile.GFile(file_name, 'wb') as file_img:
            file_img.write(file_content=img_png.numpy())


train(img_batch=img_batch, iter_nums=500)

