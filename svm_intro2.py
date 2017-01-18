import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn.svm as svm
# get_ipython().magic(u'matplotlib inline')
tf.logging.set_verbosity(tf.logging.ERROR)
plt.rcParams["figure.figsize"] = [12,7]

import numpy.random as rand

h = 0.02

# In[37]:

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
xspiral = np.row_stack((blue,red))
yspiral = np.array([1]*num_data+[2]*num_data)

x_min, x_max = xspiral[:, 0].min() - 0.1, xspiral[:, 0].max() + .1
y_min, y_max = xspiral[:, 1].min() - .1, xspiral[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# # More interesting data

# In[40]:

plt.scatter(red[:,0],red[:,1], color='red')
plt.scatter(blue[:,0],blue[:,1], color='blue')

print "This is the input data"
plt.show()

# # Linear SVM

# In[42]:

# try a linear svm
# generate linear svm
clf = svm.SVC(kernel='linear')

print "fit a linear svm"
clf.fit(xspiral, yspiral)

# predict the class for each of the points in the grid

print "generate a map of the classes"
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


# In[43]:

# plot the results as a shaded map
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(xspiral[:num_data,0],xspiral[:num_data,1], color='blue')
plt.scatter(xspiral[num_data:,0], xspiral[num_data:,1], color='red')

print "this shows the class boundaries"
plt.show()

# # Gaussian Kernel

# In[51]:

# try a Gaussian kernel svm
clf = svm.SVC(kernel='rbf', gamma=1.0)

print "Fit a Gaussian kernel RBF"
clf.fit(xspiral, yspiral)

# predict the class for each of the points in the grid
print "generate a map of the classes"
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


# In[52]:

# plot the results as a shaded map
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(xspiral[:num_data,0],xspiral[:num_data,1], color='blue')
plt.scatter(xspiral[num_data:,0], xspiral[num_data:,1], color='red')

print "this shows the class boundaries"
plt.show()

# In[ ]:



