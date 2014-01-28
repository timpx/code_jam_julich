# coding: utf-8

import theano 
from theano import tensor as T

"""
From http://deeplearning.net/software/theano/tutorial/gradients.html#computing-the-jacobian
and
http://deeplearning.net/software/theano/tutorial/profiling.html

To profile this: 
$ THEANO_FLAGS=optimizer_excluding=fusion:inplace,profile=True python theano_jacobian.py
"""

x = T.dvector('x')
y = x ** 2
J, updates = theano.scan(lambda i, y,x : T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y,x])
f = theano.function([x], J, updates=updates)
print f([4, 4])


