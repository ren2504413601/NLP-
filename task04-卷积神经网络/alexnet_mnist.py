# -*- coding=UTF-8 -*-  
'''
参考：https://baike.baidu.com/item/AlexNet/22689612
基于 TensorFlow 1.x 
'''
'''
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("datasets\\MNIST_data", one_hot=True)

# # 输入数据  
# import input_data  
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)  

# 定义网络超参数  
learning_rate = 0.001  
training_iters = 200000  
batch_size = 64  
display_step = 20  
# 定义网络参数  
n_input = 784  # 输入的维度  
n_classes = 10 # 标签的维度  
dropout = 0.8  # Dropout 的概率  
# 占位符输入  
x = tf.placeholder(tf.float32, [None, n_input])  
y = tf.placeholder(tf.float32, [None, n_classes])  
keep_prob = tf.placeholder(tf.float32)  
# 卷积操作  
def conv2d(name, l_input, w, b):  
    return tf.nn.relu(tf.nn.bias_add(  
    tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b)  
    , name=name)  
# 最大下采样操作  
def max_pool(name, l_input, k):  
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1],   
    strides=[1, k, k, 1], padding='SAME', name=name)  
# 归一化操作  
def norm(name, l_input, lsize=4):  
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)  
# 定义整个网络   
def alex_net(_X, _weights, _biases, _dropout):  
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1]) # 向量转为矩阵  
    # 卷积层  
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])  
    # 下采样层  
    pool1 = max_pool('pool1', conv1, k=2)  
    # 归一化层  
    norm1 = norm('norm1', pool1, lsize=4)  
    # Dropout  
    norm1 = tf.nn.dropout(norm1, _dropout)  
   
    # 卷积  
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])  
    # 下采样  
    pool2 = max_pool('pool2', conv2, k=2)  
    # 归一化  
    norm2 = norm('norm2', pool2, lsize=4)  
    # Dropout  
    norm2 = tf.nn.dropout(norm2, _dropout)  
   
    # 卷积  
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])  
    # 下采样  
    pool3 = max_pool('pool3', conv3, k=2)  
    # 归一化  
    norm3 = norm('norm3', pool3, lsize=4)  
    # Dropout  
    norm3 = tf.nn.dropout(norm3, _dropout)  
   
    # 全连接层，先把特征图转为向量  
    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]])   
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')   
    # 全连接层  
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') 
    # Relu activation  
    # 网络输出层  
    out = tf.matmul(dense2, _weights['out']) + _biases['out']  
    return out  
   
# 存储所有的网络参数  
weights = {  
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),  
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),  
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),  
    'wd1': tf.Variable(tf.random_normal([4*4*256, 1024])),  
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),  
    'out': tf.Variable(tf.random_normal([1024, 10]))  
}  
biases = {  
    'bc1': tf.Variable(tf.random_normal([64])),  
    'bc2': tf.Variable(tf.random_normal([128])),  
    'bc3': tf.Variable(tf.random_normal([256])),  
    'bd1': tf.Variable(tf.random_normal([1024])),  
    'bd2': tf.Variable(tf.random_normal([1024])),  
    'out': tf.Variable(tf.random_normal([n_classes]))  
}  
# 构建模型  
pred = alex_net(x, weights, biases, keep_prob)  
# 定义损失函数和学习步骤  
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  
# 测试网络  
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))  
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  
# 初始化所有的共享变量  
init = tf.initialize_all_variables()  
# 开启一个训练  
with tf.Session() as sess:  
    sess.run(init)  
    step = 1  
    # Keep training until reach max iterations  
    while step * batch_size < training_iters:  
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  
        # 获取批数据  
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})  
        if step % display_step == 0:  
            # 计算精度  
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})  
            # 计算损失值  
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})  
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)  )
        step += 1  
    print ("Optimization Finished!"  )
    # 计算测试精度  
    print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})  )
'''

'''
https://www.jianshu.com/p/a66c67594d16?utm_campaign=haruki&utm_content=note&utm_medium=reader_share&utm_source=qq
基于 tensorflow 2.x
'''  
 
import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout
from keras.optimizers import Adam


# Load oxflower17 dataset
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split
x, y = oxflower17.load_data(one_hot=True)

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle = True)



#Data augumentation with Keras tools
from keras.preprocessing.image import ImageDataGenerator
img_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

#Build AlexNet model
def AlexNet(width, height, depth, classes):
    
    model = Sequential()
    
    #First Convolution and Pooling layer
    model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(width,height,depth),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    
    #Second Convolution and Pooling layer
    model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    
    #Three Convolution layer and Pooling Layer
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    
    #Fully connection layer
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000,activation='relu'))
    model.add(Dropout(0.5))
    
    #Classfication layer
    model.add(Dense(classes,activation='softmax'))

    return model
  
AlexNet_model = AlexNet(224,224,3,17)
AlexNet_model.summary()
AlexNet_model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss = 'categorical_crossentropy',metrics=['accuracy'])

#Start training using dataaugumentation generator
History = AlexNet_model.fit_generator(img_gen.flow(X_train*255, y_train, batch_size = 16),
                                      steps_per_epoch = len(X_train)/16, validation_data = (X_test,y_test), epochs = 30 )

#Plot Loss and Accuracy
import matplotlib.pyplot as plt
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.show()
