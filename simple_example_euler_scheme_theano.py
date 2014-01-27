# x += dt*dx
from theano import *
import theano.tensor as T

x = T.dscalar()
dt = T.dscalar()
dx = T.dscalar()

result, updates = scan(fn=lambda x, dx, dt: x + dx*dt, 
							  outputs_info=T.dscalar(),
							  non_sequences=dt, 
							  n_steps = 10)
f = function(inputs=[dx, dt], outputs=result, updates=updates)

print f(3., 1., 1.)