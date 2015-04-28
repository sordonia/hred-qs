import numpy
import adam
import theano
import theano.tensor as T
from collections import OrderedDict
from theano_extensions import ProbsGrabber

def sharedX(value, name=None, borrow=False, dtype=None):
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)

def Adam(grads, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    return adam.Adam(grads, lr, b1, b2, e)

def Adagrad(grads, lr):
    """
    Taken from pylearn2, https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/training_algorithms/learning_rule.py
    """
    updates = OrderedDict()
    for param in grads.keys():
        # sum_square_grad := \sum g^2
        sum_square_grad = sharedX(param.get_value() * 0.)
        if param.name is not None:
            sum_square_grad.name = 'sum_square_grad_' + param.name

        # Accumulate gradient
        new_sum_squared_grad = sum_square_grad + T.sqr(grads[param])

        # Compute update
        delta_x_t = (- lr / T.sqrt(numpy.float32(1e-5) + new_sum_squared_grad)) * grads[param]

        # Apply update
        updates[sum_square_grad] = new_sum_squared_grad
        updates[param] = param + delta_x_t
    return updates

def Adadelta(grads, decay=0.95, epsilon=1e-6): 
    """
    Taken from pylearn2, https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/training_algorithms/learning_rule.py
    """
    updates = OrderedDict()
    for param in grads.keys():
        # mean_squared_grad := E[g^2]_{t-1}
        mean_square_grad = sharedX(param.get_value() * 0.)
        # mean_square_dx := E[(\Delta x)^2]_{t-1}
        mean_square_dx = sharedX(param.get_value() * 0.)

        if param.name is not None:
            mean_square_grad.name = 'mean_square_grad_' + param.name
            mean_square_dx.name = 'mean_square_dx_' + param.name

        # Accumulate gradient
        new_mean_squared_grad = (
            decay * mean_square_grad +
            (1 - decay) * T.sqr(grads[param])
        )

        # Compute update
        rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
        rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon) 
        delta_x_t = - rms_dx_tm1 / rms_grad_t * grads[param]

        # Accumulate updates
        new_mean_square_dx = (
            decay * mean_square_dx +
            (1 - decay) * T.sqr(delta_x_t)
        )

        # Apply update
        updates[mean_square_grad] = new_mean_squared_grad
        updates[mean_square_dx] = new_mean_square_dx
        updates[param] = param + delta_x_t

    return updates

def RMSProp(grads, lr, decay=0.95, eta=0.9, epsilon=1e-6): 
    """
    Taken from pylearn2, https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/training_algorithms/learning_rule.py
    """ 
    updates = OrderedDict()
    for param in grads.keys():
        # mean_squared_grad := E[g^2]_{t-1}
        mean_square_grad = sharedX(param.get_value() * 0.)
        mean_grad = sharedX(param.get_value() * 0.)
        delta_grad = sharedX(param.get_value() * 0.)

        if param.name is None:
            raise ValueError("Model parameters must be named.")
        
        mean_square_grad.name = 'mean_square_grad_' + param.name
        
        # Accumulate gradient
        
        new_mean_grad = (decay * mean_grad + (1 - decay) * grads[param])
        new_mean_squared_grad = (decay * mean_square_grad + (1 - decay) * T.sqr(grads[param]))

        # Compute update 
        scaled_grad = grads[param] / T.sqrt(new_mean_squared_grad - new_mean_grad ** 2 + epsilon)
        new_delta_grad = eta * delta_grad - lr * scaled_grad 

        # Apply update
        updates[delta_grad] = new_delta_grad
        updates[mean_grad] = new_mean_grad
        updates[mean_square_grad] = new_mean_squared_grad
        updates[param] = param + new_delta_grad

    return updates 

class Maxout(object):
    def __init__(self, maxout_part):
        self.maxout_part = maxout_part

    def __call__(self, x):
        shape = x.shape
        if x.ndim == 2:
            shape1 = T.cast(shape[1] / self.maxout_part, 'int64')
            shape2 = T.cast(self.maxout_part, 'int64')
            x = x.reshape([shape[0], shape1, shape2])
            x = x.max(2)
        else:
            shape1 = T.cast(shape[2] / self.maxout_part, 'int64')
            shape2 = T.cast(self.maxout_part, 'int64')
            x = x.reshape([shape[0], shape[1], shape1, shape2])
            x = x.max(3)
        return x

def UniformInit(rng, sizeX, sizeY, lb=-0.01, ub=0.01):
    """ Uniform Init """
    return rng.uniform(size=(sizeX, sizeY), low=lb, high=ub).astype(theano.config.floatX)

def OrthogonalInit(rng, shape):
    if len(shape) != 2:
        raise ValueError

    if shape[0] == shape[1]:
        # For square weight matrices we can simplify the logic
        # and be more exact:
        M = rng.randn(*shape).astype(theano.config.floatX)
        Q, R = numpy.linalg.qr(M)
        Q = Q * numpy.sign(numpy.diag(R))
        return Q

    M1 = rng.randn(shape[0], shape[0]).astype(theano.config.floatX)
    M2 = rng.randn(shape[1], shape[1]).astype(theano.config.floatX)

    # QR decomposition of matrix with entries in N(0, 1) is random
    Q1, R1 = numpy.linalg.qr(M1)
    Q2, R2 = numpy.linalg.qr(M2)
    # Correct that NumPy doesn't force diagonal of R to be non-negative
    Q1 = Q1 * numpy.sign(numpy.diag(R1))
    Q2 = Q2 * numpy.sign(numpy.diag(R2))

    n_min = min(shape[0], shape[1])
    return numpy.dot(Q1[:, :n_min], Q2[:n_min, :])

"""
def OrthogonalInit(rng, sizeX, sizeY, sparsity=-1, scale=1):
    sizeX = int(sizeX)
    sizeY = int(sizeY)

    assert sizeX == sizeY, 'for orthogonal init, sizeX == sizeY'

    if sparsity < 0:
        sparsity = sizeY
    else:
        sparsity = numpy.minimum(sizeY, sparsity)

    values = numpy.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals

    u,s,v = numpy.linalg.svd(values)
    values = u * scale
    return values.astype(theano.config.floatX)
"""

# grabber = ProbsGrabber()
def GrabProbs(class_probs, target, gRange=None):
    if class_probs.ndim > 2:
        class_probs = class_probs.reshape((class_probs.shape[0] * class_probs.shape[1], class_probs.shape[2]))
    else:
        class_probs = class_probs
    if target.ndim > 1:
        tflat = target.flatten()
    else:
        tflat = target
     
    # return grabber(class_probs, target).flatten()
    ### Hack for Theano, much faster than [x, y] indexing 
    ### avoids a copy onto the GPU
    return T.diag(class_probs.T[tflat])

def NormalInit(rng, sizeX, sizeY, scale=0.01, sparsity=-1):
    """ 
    Normal Initialization
    """
    sizeX = int(sizeX)
    sizeY = int(sizeY)
    
    if sparsity < 0:
        sparsity = sizeY
     
    sparsity = numpy.minimum(sizeY, sparsity)
    values = numpy.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals
        
    return values.astype(theano.config.floatX)

def ConvertTimedelta(seconds_diff): 
    hours = seconds_diff // 3600
    minutes = (seconds_diff % 3600) // 60
    seconds = (seconds_diff % 60)
    return hours, minutes, seconds

def SoftMax(x):
    x = T.exp(x - T.max(x, axis=x.ndim-1, keepdims=True))
    return x / T.sum(x, axis=x.ndim-1, keepdims=True)
