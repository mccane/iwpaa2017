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

# let's try some more interesting data -  a spiral
num_data = 400
theta = np.linspace(0,4*math.pi,num_data)
theta = theta + rand.normal(size=num_data,scale=.1)
r = theta + rand.normal(size=num_data,scale=0.4)
xspiral = r*np.cos(theta)
yspiral = r*np.sin(theta)
xs2 = -r*np.cos(theta)
ys2 = -r*np.sin(theta)
plt.scatter(xspiral,yspiral, color='red')
plt.scatter(xs2,ys2, color='blue')

print "spiral data"
plt.show()

blue = np.column_stack((xspiral,yspiral))
red = np.column_stack((xs2,ys2))
xspiral = np.row_stack((blue,red))
yspiral = np.array([0]*num_data+[1]*num_data)

x_min, x_max = xspiral[:, 0].min() - 0.1, xspiral[:, 0].max() + .1
y_min, y_max = xspiral[:, 1].min() - .1, xspiral[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# In[26]:

# try a linear svm
# generate linear svm

print "Learn a deep neural network"
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=2)]
clf = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                     hidden_units=[3,2],
                                     optimizer=tf.train.AdagradOptimizer(learning_rate=.1),
                                     n_classes=2)

clf.fit(xspiral, yspiral, steps=1000)

# predict the class for each of the points in the grid

print "Build the output distribution"
Z = np.array(list(clf.predict(np.c_[xx.ravel(), yy.ravel()], as_iterable=True)))
Z = Z.reshape(xx.shape)

# In[ ]:

# plot the results as a shaded map
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(xspiral[:num_data,0],xspiral[:num_data,1], color='blue')
plt.scatter(xspiral[num_data:,0], xspiral[num_data:,1], color='red')
plt.show()

# # Exercise
# - Once again this network doesn't work
# - Try updating the parameters to get it to work


