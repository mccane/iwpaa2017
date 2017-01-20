
# coding: utf-8

# # Why TensorFlow? (or Theano or Torch or ...)
# 
# - tools like tensorflow have transformed ML
# - there are two main advantages:
#     1. code can be easily deployed on a GPU
#     2. derivatives are computed automatically (automatic differentiation)
# 

# # Why TensorFlow?
# 
# - but there are costs:
#     1. the programming model is weird
#     2. it can be hard to debug
#     3. sometimes you can't do things that would be easy with "normal" programming

# # Low Level TensorFlow
# 
# - you need to learn the low-level interface to really use TensorFlow
# 
# - this can be painful

# # Normal programming
# 
# - the program code is the model
# 
# - a compiler converts the model into an executable
# 
# or
# 
# - an interpreter executes the model

# # TensorFlow programming
# 
# 1. the program code creates the model (called a Graph in tf)
#     - usually by creating variables and specifying operations on variables

# In[16]:

import tensorflow as tf

# create a variable
x = tf.Variable(0)
# initialize variables
initialize = tf.global_variables_initializer()


# # TensorFlow programming
# 
# 2. the program creates a tensorflow interpreter and asks the interpreter to execute operations

# In[17]:

sess = tf.Session()
sess.run(initialize)
print(sess.run(x))

