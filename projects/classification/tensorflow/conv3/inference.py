import tensorflow as tf
from net import simpleconv3net
import sys
import numpy as np
import cv2

testsize = 48
x = tf.placeholder(tf.float32, [1,testsize,testsize,3])
y = simpleconv3net(x)
y = tf.nn.softmax(y)

img = cv2.imread(sys.argv[2]).astype(np.float32)
img = cv2.resize(img,(testsize,testsize),interpolation=cv2.INTER_NEAREST)

imgs = np.zeros([1,testsize,testsize,3],dtype=np.float32)
imgs[0:1,] = img

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess,sys.argv[1])
result = sess.run(y, feed_dict={x:imgs})

print result
