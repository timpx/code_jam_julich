# coding: utf-8

"""
Draw samples from a Normal Distribution
"""

from sympy import Symbol
from sympy.stats.crv_types import NormalDistribution

x      = Symbol('x')
mu     = Symbol('mu', bounded=True)
sigma  = Symbol('sigma', positive=True)
result = NormalDistribution(mu, sigma)(x)


from sympy.utilities.lambdify import lambdify
fn_numpy   = lambdify([x, mu, sigma], result, 'numpy')


from sympy.printing.theanocode import theano_function
f_sym_theano = theano_function([x, mu, sigma], [result], dims = {x:1, mu:0, sigma:0})

from theano.scalar.basic_sympy import SymPyCCode
from theano.tensor.elemwise import Elemwise
normal_op = Elemwise(SymPyCCode([x, mu, sigma], result))

import theano
xt = theano.tensor.vector('x')
mut = theano.scalar.float32('mu')
sigmat = theano.scalar.float32('sigmat')

ft = theano.function([xt, mut, sigmat], normal_op(xt, mut, sigmat))


# evaluate them
import numpy
xx     = numpy.linspace(0, 1, 5)
muv    = 0.0
sigmav = 1.0


timeit f_sym_theano(xx, numpy.array([muv]),numpy.array([sigmav]))
timeit fn_numpy(xx, muv, sigmav)
timeit ft(xx, muv, sigmav)
