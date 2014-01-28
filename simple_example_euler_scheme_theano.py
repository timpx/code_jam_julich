# x += dt*dx
from theano import *
import theano.tensor as T

x = T.dscalar("x")
dt = T.dscalar("dt")

result, updates = scan(fn=lambda x, dt : x + dt, 
							  outputs_info=None,
							  non_sequences= dt,
							  n_steps = 10)

f = function(inputs=[x, dt], outputs=result, updates=updates)

print f( 2., 1.)