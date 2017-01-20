
# coding: utf-8

# In[36]:

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn.svm as svm
get_ipython().magic(u'matplotlib inline')
tf.logging.set_verbosity(tf.logging.INFO)
plt.rcParams["figure.figsize"] = [12,7]

import numpy.random as rand
h = 0.02

# let's try some more interesting data -  a spiral
num_data = 400
theta = np.linspace(0,4*math.pi,num_data)
theta = theta + rand.normal(size=num_data,scale=.1)
r = theta + rand.normal(size=num_data,scale=0.4)
xspiral = r*np.cos(theta)
yspiral = r*np.sin(theta)
xs2 = -r*np.cos(theta)
ys2 = -r*np.sin(theta)

blue = np.column_stack((xspiral,yspiral))
red = np.column_stack((xs2,ys2))
spiral_data = np.row_stack((blue,red))
spiral_labels = np.zeros((800,2))
spiral_labels[:400,0] = 1
spiral_labels[400:,1] = 1

x_min, x_max = spiral_data[:, 0].min() - 0.1, spiral_data[:, 0].max() + .1
y_min, y_max = spiral_data[:, 1].min() - .1, spiral_data[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
test_data = np.c_[xx.ravel(), yy.ravel()]

# using low level interfaces in tf


# # Going Deeper
# 
# $$
# f(x) = \sigma( W_3 \sigma( W_2 \sigma (W_1 x + b_1) + b_2 ) + b_3)
# $$

# # Layer 1

# In[37]:

# this is a variable for the input
x = tf.placeholder(tf.float32, [None, 2])
y_ = tf.placeholder(tf.float32, [None, 2])

W = tf.Variable(tf.random_normal([2, 10], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[10]))

out_layer1 = tf.nn.relu(tf.matmul(x, W) + b)


# # Layer 2 and 3

# In[38]:

# add a layer
W2 = tf.Variable(tf.random_normal([10,10], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[10]))

out_layer2 = tf.nn.relu(tf.matmul(out_layer1, W2) + b2)

# add a layer
W3 = tf.Variable(tf.random_normal([10,2], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[2]))

out_layer3 = tf.nn.relu(tf.matmul(out_layer2, W3) + b3)


# # Loss Function

# In[39]:

# define loss function
# this is the cross entropy loss function
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out_layer3))


# # Operators

# In[40]:

# specify the optimizer and the operation causing the optimizer to step
train_step = tf.train.AdagradOptimizer(0.1).minimize(cross_entropy)

# an operation to initialize variables
initialise = tf.global_variables_initializer()

# operators for computing accuracy
correct_prediction = tf.equal(tf.argmax(out_layer3, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# # Create session and run

# In[41]:

# create a session
sess = tf.Session()
sess.run(initialise)

# run the training
for i in xrange(10000):
    sess.run(train_step, feed_dict = {x: spiral_data, y_: spiral_labels})
    if i%100:
        print(sess.run(accuracy, feed_dict={x: spiral_data,
                                            y_: spiral_labels}))


# In[42]:

Z = sess.run(tf.argmax(out_layer3, 1), feed_dict={x: test_data})
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(spiral_data[:num_data,0],spiral_data[:num_data,1], color='blue')
plt.scatter(spiral_data[num_data:,0], spiral_data[num_data:,1], color='red')

