
# coding: utf-8

# In[3]:

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


# # A one layer network
# 
# $$
# f(x) = \frac{1}{1+e^{-(W x + b)}}
# $$

# In[4]:

# this is a variable for the input
x = tf.placeholder(tf.float32, [None, 2])

W = tf.Variable(tf.zeros([2, 2]))
b = tf.Variable(tf.zeros([2]))
out_layer1 = tf.matmul(x, W) + b

# placeholder for training data labels
# 2 classes, so 2 output nodes
y_ = tf.placeholder(tf.float32, [None, 2])


# # The loss function
# 
# $$
# J(W,b) = - \sum_{i=1}^N y_i \log(f(x_i)) + (1-y_i) \log(1-f(x_i))
# $$

# In[5]:

# define loss function

# this is the cross entropy loss function
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out_layer1))


# # Define the optimizer

# In[6]:

# specify the optimizer and the operation causing the optimizer to step
train_step = tf.train.AdagradOptimizer(1.0).minimize(cross_entropy)

# an operation to initialize variables
initialise = tf.global_variables_initializer()

# operators for computing accuracy
correct_prediction = tf.equal(tf.argmax(out_layer1, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# # Create session and execute

# In[7]:

# create a session
sess = tf.Session()
sess.run(initialise)

# run the training
for i in xrange(1000):
    sess.run(train_step, feed_dict = {x: spiral_data, y_: spiral_labels})
    print(sess.run(accuracy, feed_dict={x: spiral_data,
                                        y_: spiral_labels}))


# In[8]:

Z = sess.run(tf.argmax(out_layer1, 1), feed_dict={x: test_data})
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(spiral_data[:num_data,0],spiral_data[:num_data,1], color='blue')
plt.scatter(spiral_data[num_data:,0], spiral_data[num_data:,1], color='red')

