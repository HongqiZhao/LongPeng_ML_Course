#http://cwiki.apachecn.org/pages/viewpage.action?pageId=10029443

from dataset import *
from net import simpleconv3net
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
txtfile = sys.argv[1]
batch_size = 1
num_classes = 2
image_size = (48,48)
learning_rate = 0.001
import numpy as np
import cv2

if __name__=="__main__":
    dataset = ImageData(txtfile,batch_size,num_classes,image_size)
    
    iterator = dataset.data.make_one_shot_iterator()
    dataset_size = dataset.dataset_size
    
    print "dataset type=",type(dataset)
    print "iterator=",type(iterator)
    print iterator
    one_element = iterator.get_next()
    print "one_element=",one_element
   
    Ylogits = simpleconv3net(one_element[0])
    print "Ylogits size=",Ylogits.shape
    Y = tf.nn.softmax(Ylogits)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=one_element[1])
    cross_entropy = tf.reduce_mean(cross_entropy)*100
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(one_element[1], 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    print "accuracy=",accuracy


    with tf.Session() as sess:
        for _ in range(1):
            result = sess.run([one_element])
            print result[0][0].shape
            print result[0][1].shape
            imagedebug = result[0][0].copy()
            labeldebug = result[0][1].copy()

    imagedebug = np.squeeze(imagedebug)
    print imagedebug
    print np.max(imagedebug)
    print labeldebug

    imagedebug = cv2.cvtColor((imagedebug*255).astype(np.uint8),cv2.COLOR_RGB2BGR)

    cv2.namedWindow("debug image",0)
    cv2.imshow("debug image",imagedebug)
    cv2.waitKey(0)



