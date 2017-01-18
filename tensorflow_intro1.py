
# coding: utf-8

# In[42]:

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn.svm as svm
# get_ipython().magic(u'matplotlib inline')
tf.logging.set_verbosity(tf.logging.INFO)
plt.rcParams["figure.figsize"] = [12,7]

import numpy.random as rand
h = 0.02

# generate linear 2d data and plot
def gen_linear(num_points):
    X = rand.rand(num_points*2,2)
    y = np.array([0]*num_points+[1]*num_points)
    for i in xrange(num_points):
        X[i,1] = X[i,0]-0.4+rand.normal(scale=0.1)
        X[num_points+i,1] = X[num_points+i,0]+0.4+rand.normal(scale=0.1)
        
    return X, y

xlin, ylin = gen_linear(200)
# also create a mesh to do contour plots
# create a mesh to plot in
x_min, x_max = xlin[:, 0].min() - 0.1, xlin[:, 0].max() + .1
y_min, y_max = xlin[:, 1].min() - .1, xlin[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# # Deep Networks
# 
# $$
# f(y) = \sigma(w_{4l} \sum_k \sigma(w_{3k} \sum_j \sigma(w_{2j} \sum_i \sigma(w_{1i} y + b_{1i}) + b_{2j}) + b_{3k}) + b_{4l})
# $$
# 
# - $\sigma$ is a non-linear (but simple) activation function
# - most commonly these days a Rectified Linear Unit (RELU):
# $$
# \sigma(y) = \max(0,y)
# $$


# # Simple Neural Network with TensorFlow

# In[56]:

# generate Deep Neural network
# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=2)]

clf = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                     hidden_units=[2,2],
                                     optimizer=tf.train.AdagradOptimizer(learning_rate=0.01),
                                     n_classes=2)

print "Fitting a neural network"
clf.fit(xlin, ylin, steps=1000)
accuracy_score = clf.evaluate(x=xlin,y=ylin)["accuracy"]

print "training accuracy = ", accuracy_score

# predict the class for each of the points in the grid

print "Building output distribution"
Z = np.array(list(clf.predict(np.c_[xx.ravel(), yy.ravel()], as_iterable=True)))
Z = Z.reshape(xx.shape)


# In[57]:

# plot the results as a shaded map
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(xlin[:200,0],xlin[:200,1], color='blue')
plt.scatter(xlin[200:400,0], xlin[200:400,1], color='red')

print "Displaying output distribution"
plt.show()

# # Exercise
# 
# - The above network often doesn't work.
# - See if you can get it to work.
# - Things to try include:
#     - changing the number of steps
#     - changing the architecture (adding more neurons or more layers)
#     - changing the learning rate

