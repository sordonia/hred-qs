# -*- coding: utf-8 -*-
import numpy
import theano
from theano.gradient import DisconnectedType
from theano.gradient import grad_undefined
from theano.gof import local_optimizer
from theano import Op, Apply
from theano.tensor.sort import ArgSortOp
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.basic import as_tensor_variable
import theano.tensor as T

from theano.sandbox.cuda import cuda_available, GpuOp
if cuda_available:
    from theano.sandbox.cuda import CudaNdarrayType
    from theano.sandbox.cuda.basic_ops import host_from_gpu, gpu_from_host
    from theano.sandbox.cuda.opt import register_opt

class MRG_RandomStreams2(MRG_RandomStreams):
    """Module component with similar interface to numpy.random
    (numpy.random.RandomState)
    """
    def __init__(self, seed=12345, use_cuda=None):
        super(MRG_RandomStreams2, self).__init__(seed, use_cuda)

    def multinomial(self, size=None, n=1, pvals=None, ndim=None, dtype='int64', nstreams=None):
        if pvals is None:
            raise TypeError('You have to specify pvals')
        pvals = as_tensor_variable(pvals)
        if size is not None:
            if any([isinstance(i, int) and i <= 0 for i in size]):
                raise ValueError('The specified size contains a dimension with value <= 0', size)

        if n == 1 and pvals.ndim == 1:
            if ndim is not None:
                raise ValueError('Provided an ndim argument to ' +
                        'MRG_RandomStreams2.multinomial, which does not use ' +
                        'the ndim argument.')
            unis = self.uniform(size=size, ndim=2, nstreams=nstreams)
            op = MultinomialFromUniform2(dtype)
            return op(pvals, unis)
        else:
            raise NotImplementedError('MRG_RandomStreams2.multinomial only ' +
                ' implemented with n == 1 and pvals.ndim = 2')

class MultinomialFromUniform2(Op):
    '''Converts samples from a uniform into sample from a multinomial.'''
    def __init__(self, odtype):
        self.odtype = odtype

    def __eq__(self, other):
        return type(self) == type(other) and self.odtype == other.odtype

    def __hash__(self):
        return hash((type(self), self.odtype))

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.odtype)

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        try:
            self.odtype
        except AttributeError:
            self.odtype = 'auto'

    def make_node(self, pvals, unis):
        pvals = T.as_tensor_variable(pvals)
        unis = T.as_tensor_variable(unis)
        if unis.ndim != 2:
            raise NotImplementedError('unis ndim should be 1', unis.ndim)
        if self.odtype == 'auto':
            odtype = pvals.dtype
        else:
            odtype = self.odtype
        out = T.tensor(dtype=odtype, broadcastable=unis.type.broadcastable)
        return Apply(self, [pvals, unis], [out])

    def grad(self, ins, outgrads):
        pvals, unis = ins
        (gz,) = outgrads
        return [T.zeros_like(x) for x in ins]

    def c_code_cache_version(self):
        return (5,)

    def c_code(self, node, name, ins, outs, sub):
        (pvals, unis) = ins
        (z,) = outs

        if self.odtype == 'auto':
            t = "PyArray_TYPE((PyArrayObject*) py_%(pvals)s)" % locals()
        else:
            t = theano.scalar.Scalar(self.odtype).dtype_specs()[1]
            if t.startswith('theano_complex'):
                t = t.replace('theano_complex', 'NPY_COMPLEX')
            else:
                t = t.upper()

        fail = sub['fail']
        return """
        if (PyArray_NDIM(%(pvals)s) != 1)
        {
            PyErr_Format(PyExc_TypeError, "pvals wrong rank");
            %(fail)s;
        }
        if (PyArray_NDIM(%(unis)s) != 2)
        {
            PyErr_Format(PyExc_TypeError, "unis wrong rank");
            %(fail)s;
        }

        if ((NULL == %(z)s)
            || ((PyArray_DIMS(%(z)s))[0] != (PyArray_DIMS(%(unis)s))[0])
            || ((PyArray_DIMS(%(z)s))[1] != (PyArray_DIMS(%(unis)s))[1])
        )
        {
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*) PyArray_ZEROS(2,
                PyArray_DIMS(%(unis)s),
                %(t)s,
                0);
            
            if (!%(z)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
                %(fail)s;
            }
        }
        
        {  
        // NESTED SCOPE
        const int nb_outcomes = PyArray_DIMS(%(pvals)s)[0];
        const int nb_rows = PyArray_DIMS(%(unis)s)[0];
        const int nb_cols = PyArray_DIMS(%(unis)s)[1];
        
        //
        // For each multinomial, loop over each possible outcome
        //        
        for (int row = 0; row < nb_rows; ++row)
        {
            for (int col = 0; col < nb_cols; ++col) {
                           
                dtype_%(pvals)s cummul = 0.;
                const dtype_%(unis)s* unis_n = (dtype_%(unis)s*)PyArray_GETPTR2(%(unis)s, row, col);
                dtype_%(z)s* z_nm = (dtype_%(z)s*) PyArray_GETPTR2(%(z)s, row, col);
                
                *z_nm = -1;
                int m = 0;
                for (m = 0; m < nb_outcomes; ++m)
                {
                    const dtype_%(pvals)s* pvals_m = (dtype_%(pvals)s*)PyArray_GETPTR1(%(pvals)s, m);
                    cummul += *pvals_m;
                    if (cummul > *unis_n)
                    {
                        *z_nm = m;
                    }
                }
                
                if (m < nb_outcomes)
                    *z_nm = m;
            }
        } 
        } // END NESTED SCOPE
        """ % locals()

         
    def perform(self, node, ins, outs):
        (pvals, unis) = ins
        (z,) = outs

        if z[0] is None or z[0].shape != numpy.sum(unis.shape):
            z[0] = numpy.zeros(unis.shape, dtype=node.outputs[0].dtype)

        z[0][:, :] = -1

        nb_outcomes = pvals.shape[0]

        for row in xrange(unis.shape[0]):
            for col in xrange(unis.shape[1]):
                cummul = 0
                unis_n = unis[row, col]

                for m in range(nb_outcomes):
                    cummul += pvals[m]

                    if cummul > unis_n:
                        z[0][row, col] = m
                        break

                # If we reached the end, use the last value.
                # If we have a real distribution [0,1], than this should never
                # happen, right? I got a segmentation fault when removing it.
                # 2014-04-08
                # This might happen due to rounding errors. 2014-05-01
                if z[0][row, col] == -1:
                    z[0][row, col] = nb_outcomes - 1;

class GpuMultinomialFromUniform2(MultinomialFromUniform2, GpuOp):
    def make_node(self, pvals, unis):
        assert pvals.dtype == 'float32'
        assert unis.dtype == 'float32'
        if not isinstance(pvals.type, CudaNdarrayType):
            raise TypeError('pvals must be cudandarray', pvals)
        if not isinstance(unis.type, CudaNdarrayType):
            raise TypeError('unis must be cudandarray', unis)       
        if self.odtype == 'auto':
            odtype = pvals.dtype
        else:
            odtype = self.odtype
        if odtype != pvals.dtype:
            raise NotImplementedError('GpuMultinomialFromUniform2 works only if'
                'self.odtype == pvals.dtype', odtype, pvals.dtype)
        br = (unis.broadcastable[0], unis.broadcastable[1])
        out = CudaNdarrayType(broadcastable=br)()
        return Apply(self, [pvals, unis], [out])

    def perform(self, node, ins, outs):
        #The perform from parent don't work with CudaNdarray.  We
        #don't need it as DebugMode will test again it as an
        #optimization insert the GPU op.
        return Op.perform(self, node, ins, outs)

    def c_code_cache_version(self):
        return (8,)

    def c_support_code_apply(self, node, nodename):
        return """
        static __global__ void k_multi_warp_%(nodename)s(
            const int nb_multi,
            const int nb_outcomes,
            float * global_pvals,
            const int pvals_stride,
            float * global_unis,
            const int unis_row_stride,
            const int unis_col_stride,
            float * global_outs,
            const int outs_row_stride,
            const int outs_col_stride
        )
        {
            // each thread takes care of one multinomial draw
            int load_per_thread = (nb_multi + blockDim.x - 1) / blockDim.x;
            int start = threadIdx.x * load_per_thread;
            int end = min(nb_multi, load_per_thread * (threadIdx.x + 1));

            for (int n=start; n<end; ++n)
            {
                float cummul = 0.;
                global_outs[n] = -1;

                bool done = false;
                const float unis_n = global_unis[n];
                
                int m = 0;
                for (m = 0; m < nb_outcomes; ++m)
                {
                    cummul += global_pvals[m];
                    if (cummul > unis_n)
                    {
                        break;
                    }
                }
                
                if (m < nb_outcomes)
                    global_outs[n] = m; 
            }
        }
        """ % locals()

    def c_code(self, node, name, ins, outs, sub):
        (pvals, unis) = ins
        (z,) = outs

        fail = sub['fail']
        return """
        
        if (CudaNdarray_NDIM(%(pvals)s) != 1)
        {
            PyErr_Format(PyExc_TypeError, "pvals wrong rank");
            %(fail)s;
        }

        if (CudaNdarray_NDIM(%(unis)s) != 2)
        {
            PyErr_Format(PyExc_TypeError, "unis wrong rank");
            %(fail)s;
        }

        if ((NULL == %(z)s)
            || (CudaNdarray_HOST_DIMS(%(z)s)[0] != CudaNdarray_HOST_DIMS(%(unis)s)[0])
            || (CudaNdarray_HOST_DIMS(%(z)s)[1] != CudaNdarray_HOST_DIMS(%(unis)s)[1]))
        {
            Py_XDECREF(%(z)s);
            npy_intp dims[2];
            dims[0] = (CudaNdarray_HOST_DIMS(%(unis)s)[0]);
            dims[1] = (CudaNdarray_HOST_DIMS(%(unis)s)[1]);
            %(z)s = (CudaNdarray*) CudaNdarray_NewDims(2, dims);
            if (!%(z)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
                %(fail)s;
            }
        }
        
        { 
            // NESTED SCOPE
            int nb_rows = CudaNdarray_HOST_DIMS(%(unis)s)[0];
            int nb_cols = CudaNdarray_HOST_DIMS(%(unis)s)[1];
            int nb_multi = nb_rows*nb_cols;
            int nb_outcomes = CudaNdarray_HOST_DIMS(%(pvals)s)[0];
            
            int max_nb_blocks = 2<<15 - 1;
            int nb_blocks = max_nb_blocks + 1;
            int nb_threads=16; // so it really starts at 32, because of the *2
            
            do
            {
                nb_threads*=2;
                if (nb_multi %% nb_threads == 0)
                    nb_blocks = nb_multi/nb_threads;
                else
                    nb_blocks = (int)((float)nb_multi/(float)nb_threads + 1.);
            } while (nb_blocks > max_nb_blocks);

            // printf("\\nN=%%i b=%%i t=%%i t*b=%%i", nb_multi, nb_blocks, nb_threads, nb_blocks*nb_threads);

            // TODO : next line is a bit hardcoded...
            if (nb_threads > 512)
            {
                // Each thread should handle more than one element
                nb_threads = 512;
            }
            
            dim3 n_blocks(nb_blocks,1,1);
            dim3 n_threads(nb_threads,1,1);
            int n_shared = 0;

            k_multi_warp_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                nb_multi,
                nb_outcomes,
                CudaNdarray_DEV_DATA(%(pvals)s),
                CudaNdarray_HOST_STRIDES(%(pvals)s)[0],
                CudaNdarray_DEV_DATA(%(unis)s),
                CudaNdarray_HOST_STRIDES(%(unis)s)[0],
                CudaNdarray_HOST_STRIDES(%(unis)s)[1],
                CudaNdarray_DEV_DATA(%(z)s),
                CudaNdarray_HOST_STRIDES(%(z)s)[0],
                CudaNdarray_HOST_STRIDES(%(z)s)[1]
            );
            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i; shared: %%i)\\n",
                    "k_multi_warp_%(name)s",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z,
                    n_shared);
                %(fail)s;
            } 
        } // END NESTED SCOPE
        """ % locals()

class KArgmax(Op):
    def __init__(self, K):
        self.K = K
        self.odtype = 'auto'

    def __eq__(self, other):
        return type(self) == type(other) 

    def __hash__(self):
        return hash((type(self), self.odtype))

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.odtype)

    def __setstate__(self, dct):
        self.__dict__.update(dct)

    def make_node(self, pvals):
        pvals = T.as_tensor_variable(pvals)
        if self.odtype == 'auto':
            odtype = pvals.dtype
        vals = T.tensor(dtype=odtype, broadcastable=pvals.type.broadcastable)
        indx = T.tensor(dtype='int32', broadcastable=pvals.type.broadcastable)
        return Apply(self, [pvals,], [vals, indx])

    def c_code_cache_version(self):
        return (8,)
    
    def c_code(self, node, name, ins, outs, sub):
        (pvals,) = ins
        (vals, indx) = outs
        k = self.K
        t = "PyArray_TYPE((PyArrayObject*) py_%(pvals)s)" % locals()
         
        fail = sub['fail']
        return """
        if (PyArray_NDIM(%(pvals)s) != 2)
        {
            PyErr_Format(PyExc_TypeError, "pvals wrong rank");
            %(fail)s;
        }

        if ((NULL == %(vals)s)
            || ((PyArray_DIMS(%(vals)s))[0] != (PyArray_DIMS(%(pvals)s))[0])
            || ((PyArray_DIMS(%(vals)s))[1] != %(k)s)
        )
        {
            Py_XDECREF(%(vals)s);
            npy_intp dimensions[] = { PyArray_DIMS(%(pvals)s)[0], %(k)s };
            %(vals)s = (PyArrayObject*) PyArray_ZEROS(2,
                dimensions,
                %(t)s,
                0);
            
            %(indx)s = (PyArrayObject*) PyArray_ZEROS(2,
                dimensions,
                %(t)s,
                0);

            if (!%(vals)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc output");
                %(fail)s;
            }
        }
        """ % locals()
 
    def perform(self, node, ins, outs):
        (pvals,) = ins
        (vals, indx, ) = outs

        if vals[0] is None or vals[0].shape != (pvals.shape[0], self.K): 
            vals[0] = numpy.zeros((pvals.shape[0], self.K), dtype=node.outputs[0].dtype)
        if indx[0] is None or indx[0].shape != (pvals.shape[0], self.K): 
            indx[0] = numpy.zeros((pvals.shape[0], self.K), dtype=node.outputs[1].dtype)

        vals[0][:, :] = -1
        indx[0][:, :] = -1
        for row in xrange(pvals.shape[0]):
            indx[0][row, :] = numpy.argsort(-pvals[row])[:self.K]
            vals[0][row, :] = pvals[row][indx[0][row]]

class GpuKArgmax(KArgmax, GpuOp):
    """
    This class is a wrapper for numpy argsort function
    """
    def __init__(self, K=1):
        KArgmax.__init__(self, K)

    def __eq__(self, other):
        return (type(self) == type(other) and self.K == other.K)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.K)

    def __str__(self):
        return (self.__class__.__name__) 

    def make_node(self, pvals):
        assert pvals.dtype == 'float32'
        if not isinstance(pvals.type, CudaNdarrayType):
            raise TypeError('pvals must be cudandarray', pvals)
        if self.odtype == 'auto':
            odtype = pvals.dtype
        else:
            odtype = self.odtype
        if odtype != pvals.dtype:
            raise NotImplementedError('GpuKArgmax works only if'
                'self.odtype == pvals.dtype', odtype, pvals.dtype)
        
        br = (pvals.broadcastable[0], pvals.broadcastable[1])
        vals = CudaNdarrayType(broadcastable=br)()
        indx = CudaNdarrayType(broadcastable=br)()
        return Apply(self, [pvals], [vals, indx])

    def perform(self, node, ins, outs):
        return Op.perform(node, ins, outs)
     
    def c_code_cache_version(self):
        return (8,)

    def c_support_code_apply(self, node, nodename):
        return """
        __global__ void extract_top_k_%(nodename)s(
            int nb_rows,
            int nb_cols,
            int k,
            float * data,
            float * values,
            float * indices) {
                
                int n = threadIdx.x;
                if (n > nb_rows) return;
                
                int offset = threadIdx.x * nb_cols;
                int end = offset + nb_cols;

                for (int x=0;x<k;++x)
                {
                    values[n*k + x] = data[offset+x];
                    indices[n*k + x] = x;
                }

                for (int x=k; x<nb_cols; ++x)
                {
                    int min_idx=n*k;
                    for (int i=0;i<k;++i)
                    {   
                        if (values[n*k+i]<values[min_idx])
                        {
                            min_idx=n*k+i; 
                        }
                    }

                    if (data[offset+x] > values[min_idx])
                    {
                        values[min_idx] = data[offset+x];
                        indices[min_idx] = x;
                    }
                }
            }
        
        """ % locals()

    def c_code(self, node, name, ins, outs, sub):
        (pvals,) = ins
        (vals,indx) = outs
        k = self.K 
        fail = sub['fail']
        return """
        if (CudaNdarray_NDIM(%(pvals)s) != 2)
        {
            PyErr_Format(PyExc_TypeError, "pvals wrong rank");
            %(fail)s;
        }

        if ((NULL == %(vals)s)
            || (CudaNdarray_HOST_DIMS(%(vals)s)[0] != CudaNdarray_HOST_DIMS(%(pvals)s)[0])
            || (CudaNdarray_HOST_DIMS(%(vals)s)[1] != %(k)s))
        {
            Py_XDECREF(%(vals)s);
            Py_XDECREF(%(indx)s);

            npy_intp dims[2];
            dims[0] = (CudaNdarray_HOST_DIMS(%(pvals)s)[0]);
            dims[1] = %(k)s; 
            %(vals)s = (CudaNdarray*) CudaNdarray_NewDims(2, dims);
            %(indx)s = (CudaNdarray*) CudaNdarray_NewDims(2, dims);
            
            if (!%(vals)s || !%(indx)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc output");
                %(fail)s;
            }
        }
         
        { 
            // NESTED SCOPE
            int nb_rows = CudaNdarray_HOST_DIMS(%(pvals)s)[0];
            int nb_cols = CudaNdarray_HOST_DIMS(%(pvals)s)[1];
            npy_intp pk = %(k)s;

            dim3 n_blocks(1,1,1);
            dim3 n_threads(nb_rows,1,1);
            int n_shared = nb_rows * nb_cols;

            extract_top_k_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                nb_rows,
                nb_cols,
                pk,
                CudaNdarray_DEV_DATA(%(pvals)s),
                CudaNdarray_DEV_DATA(%(vals)s),
                CudaNdarray_DEV_DATA(%(indx)s)
            );
            
            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i; shared: %%i)\\n",
                    "k_multi_warp_%(name)s",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z,
                    n_shared);
                %(fail)s;
            } 
        } // END NESTED SCOPE
        """ % locals()

    def grad(self, inputs, output_grads):
        pass
    
    def R_op(self, inputs, eval_points):
        pass

class Assigner(Op):
    def __init__(self):
        self.odtype = 'auto'

    def __eq__(self, other):
        return type(self) == type(other) 

    def __hash__(self):
        return hash((type(self), self.odtype))

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.odtype)

    def __setstate__(self, dct):
        self.__dict__.update(dct)

    def make_node(self,pvals,indx,gr):
        pvals = T.as_tensor_variable(pvals)
        indx = T.as_tensor_variable(indx)
        gr = T.as_tensor_variable(gr)
        if self.odtype == 'auto':
            odtype = pvals.dtype
        vals = T.tensor(dtype=odtype, broadcastable=pvals.type.broadcastable)
        return Apply(self, [pvals,indx,gr], [vals])

    def c_code_cache_version(self):
        return (8,)
    
    def c_code(self, node, name, ins, outs, sub):
        (pvals,indx,gr,) = ins
        (vals,) = outs
        t = "PyArray_TYPE((PyArrayObject*) py_%(pvals)s)" % locals()
         
        fail = sub['fail']
        return """
        if ((NULL == %(vals)s)
            || ((PyArray_DIMS(%(vals)s))[0] != (PyArray_DIMS(%(pvals)s))[0]))
        {
            Py_XDECREF(%(vals)s);
            npy_intp dimensions[] = { PyArray_DIMS(%(pvals)s)[0], PyArray_DIMS(%(pvals)s)[1] };
            %(vals)s = (PyArrayObject*) PyArray_ZEROS(2,
                dimensions,
                %(t)s,
                0);
            
            if (!%(vals)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc output");
                %(fail)s;
            }
        }
        """ % locals()
 
    def perform(self, node, ins, outs):
        (pvals,indx,gr,) = ins
        (vals,) = outs
        
        vals[0] = numpy.zeros(pvals.shape, dtype=node.outputs[0].dtype)
         
        for row in range(indx.shape[0]):
            for col in range(indx.shape[1]):
                vals[0][row,col,indx[row,col]] = gr[row,col] 

class ProbsGrabber(Op):
    def __init__(self):
        self.odtype = 'auto'

    def __eq__(self, other):
        return type(self) == type(other) 

    def __hash__(self):
        return hash((type(self), self.odtype))

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.odtype)

    def __setstate__(self, dct):
        self.__dict__.update(dct)

    def make_node(self, pvals, indx):
        pvals = T.as_tensor_variable(pvals)
        indx = T.as_tensor_variable(indx)
        if self.odtype == 'auto':
            odtype = pvals.dtype
        vals = T.tensor(dtype=odtype, broadcastable=(pvals.broadcastable[0], pvals.broadcastable[1]))
        return Apply(self, [pvals,indx], [vals,])

    def c_code_cache_version(self):
        return ()
    
    def grad(self,inp,grads):
        x, indx, = inp
        gz, = grads
        return [Assigner()(x,indx,gz), grad_undefined(self,1,inp[1])] 

    def c_code(self, node, name, ins, outs, sub):
        (pvals,indx,) = ins
        (vals,) = outs
        t = "PyArray_TYPE((PyArrayObject*) py_%(pvals)s)" % locals()
         
        fail = sub['fail']
        return """
        if ((NULL == %(vals)s)
            || ((PyArray_DIMS(%(vals)s))[0] != (PyArray_DIMS(%(pvals)s))[0]))
        {
            Py_XDECREF(%(vals)s);
            npy_intp dimensions[] = { PyArray_DIMS(%(pvals)s)[0], PyArray_DIMS(%(pvals)s)[1] };
            %(vals)s = (PyArrayObject*) PyArray_ZEROS(2,
                dimensions,
                %(t)s,
                0);
            
            if (!%(vals)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc output");
                %(fail)s;
            }
        }
        """ % locals()
 
    def perform(self, node, ins, outs):
        (pvals,indx,) = ins
        (vals,) = outs
        
        vals[0] = numpy.zeros(indx.shape, dtype=node.outputs[0].dtype)
        
        for row in range(indx.shape[0]):
            for col in range(indx.shape[1]):
                vals[0][row,col] = pvals[row,col,indx[row,col]]

class GpuAssigner(Assigner, GpuOp):
    def __init__(self):
        self.odtype = 'auto'
        self.grads = CudaNdarrayType()()

    def __eq__(self, other):
        return type(self) == type(other) 

    def __hash__(self):
        return hash((type(self), self.odtype))

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.odtype)

    def __setstate__(self, dct):
        self.__dict__.update(dct)

    def make_node(self,pvals,indx,gr):
        assert pvals.dtype == 'float32'
        if not isinstance(pvals.type, CudaNdarrayType):
            raise TypeError('pvals must be cudandarray', pvals)
        if self.odtype == 'auto':
            odtype = pvals.dtype
        else:
            odtype = self.odtype
        if odtype != pvals.dtype:
            raise NotImplementedError('GpuAssigner works only if'
                'self.odtype == pvals.dtype', odtype, pvals.dtype)
        
        indx = T.as_tensor_variable(indx)
        if self.odtype == 'auto':
            odtype = pvals.dtype
        # vals = CudaNdarrayType(broadcastable=pvals.type.broadcastable)()
        return Apply(self, [pvals,indx,gr], [self.grads])

    def perform(self, node, ins, outs):
        return Op.perform(node, ins, outs)
     
    def c_code_cache_version(self):
        return (5,)
   
    def c_support_code_apply(self, node, nodename):
        return """
        __global__ void put_probs_%(nodename)s(
            int nb_rows,
            int nb_cols,
            int nb_dims,
            float * vals,
            float * gr,
            const int * indices) {
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid >= nb_rows*nb_cols) return; 
                vals[tid*nb_dims + indices[tid]] = gr[tid];
            } 
        """ % locals()

    def c_code(self, node, name, ins, outs, sub):
        (pvals,indx,gr,) = ins
        (vals,) = outs
        fail = sub['fail']
        return """
        if (!CudaNdarray_is_c_contiguous(%(pvals)s))
        {
            PyErr_SetString(PyExc_MemoryError, "Must be C contiguous");
            %(fail)s;
        }

        if ((NULL == %(vals)s)
            || (CudaNdarray_HOST_DIMS(%(vals)s)[0] != CudaNdarray_HOST_DIMS(%(pvals)s)[0])
            || (CudaNdarray_HOST_DIMS(%(vals)s)[1] != CudaNdarray_HOST_DIMS(%(pvals)s)[1]))
        {
            Py_XDECREF(%(vals)s);
            
            npy_intp dims[3];
            dims[0] = (CudaNdarray_HOST_DIMS(%(pvals)s)[0]);
            dims[1] = (CudaNdarray_HOST_DIMS(%(pvals)s)[1]);
            dims[2] = (CudaNdarray_HOST_DIMS(%(pvals)s)[2]);
            
            %(vals)s = (CudaNdarray *) CudaNdarray_NewDims(3, dims);
            cudaError_t err = cudaMemset(%(vals)s->devdata, 0, CudaNdarray_SIZE(%(vals)s)*4);
            if (cudaSuccess != err)
            {
                PyErr_Format(PyExc_MemoryError,
                             "GpuAssigner: Error memsetting %%ld"
                             " bytes of device memory. %%s",
                             (long)(CudaNdarray_SIZE(%(vals)s) * 4),
                             cudaGetErrorString(err));
                Py_XDECREF(%(vals)s);
                %(vals)s = NULL;
                %(fail)s;
            }
             
            if (!%(vals)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc output");
                %(fail)s;
            }
        }
         
        { 
            // NESTED SCOPE
            int nb_rows = CudaNdarray_HOST_DIMS(%(pvals)s)[0];
            int nb_cols = CudaNdarray_HOST_DIMS(%(pvals)s)[1];
            int nb_dims = CudaNdarray_HOST_DIMS(%(pvals)s)[2]; 
            int n_blocks = (nb_rows*nb_cols + 512 - 1) / 512; 
             
            PyArrayObject *cpu_indices_arr = PyArray_GETCONTIGUOUS(%(indx)s);
     		int* d_indices_arr = (int*)device_malloc(PyArray_NBYTES(cpu_indices_arr));

            if(!d_indices_arr)
                return -1;

     		cudaError_t err = cudaMemcpy(d_indices_arr,
                                         PyArray_DATA(cpu_indices_arr),
                                         PyArray_NBYTES(cpu_indices_arr),
                                         cudaMemcpyHostToDevice);
            
            if(err != cudaSuccess){
                PyErr_Format(
                    PyExc_RuntimeError,
                    "GpuProbsGrabber: cudaMemcpy returned an error: %%s",
                    cudaGetErrorString(err));
                return -1;
            }
            
            put_probs_%(name)s<<<n_blocks, 512>>>(
                nb_rows,
                nb_cols,
                nb_dims,
                CudaNdarray_DEV_DATA(%(vals)s),
                CudaNdarray_DEV_DATA(%(gr)s),
                d_indices_arr
            );
            
            device_free(d_indices_arr);
            Py_XDECREF(cpu_indices_arr);

            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                %(fail)s;
            } 
        }
        // END NESTED SCOPE
        """ % locals()

class GpuProbsGrabber(ProbsGrabber, GpuOp):
    """
    This class is a wrapper for numpy argsort function
    """
    def __init__(self):
        ProbsGrabber.__init__(self)

    def __eq__(self, other):
        return (type(self) == type(other)) 

    def __hash__(self):
        return hash(type(self)) 

    def __str__(self):
        return (self.__class__.__name__) 

    def make_node(self, pvals, indx):
        assert pvals.dtype == 'float32'
        if not isinstance(pvals.type, CudaNdarrayType):
            raise TypeError('pvals must be cudandarray', pvals)
        if self.odtype == 'auto':
            odtype = pvals.dtype
        else:
            odtype = self.odtype
        if odtype != pvals.dtype:
            raise NotImplementedError('GpuProbsGrabber works only if'
                'self.odtype == pvals.dtype', odtype, pvals.dtype)
        
        indx = T.as_tensor_variable(indx)
        br = (pvals.broadcastable[0], pvals.broadcastable[1])
        vals = CudaNdarrayType(broadcastable=br)()
        return Apply(self, [pvals,indx],[vals])

    def perform(self, node, ins, outs):
        return Op.perform(node, ins, outs)
     
    def c_code_cache_version(self):
        return (5,)

    def grad(self,inp,grads):
        x,indx,=inp
        gz, = grads
        return [GpuAssigner()(x,indx,gz), grad_undefined(self,1,inp[1])] 

    def c_support_code_apply(self, node, nodename):
        return """
        __global__ void grab_probs_%(nodename)s(
            int nb_rows,
            int nb_cols,
            int nb_dims,
            float * data,
            float * vals,
            const int * indices) {
                // 3 x 5 x 10
                // tid = 4 (2 row, 1 col)
                // 4 * 10 elements
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid >= nb_rows*nb_cols) return;
                // vals[tid] = indices[tid];
                vals[tid] = data[tid*nb_dims + indices[tid]];
            } 
        """ % locals()

    def c_code(self, node, name, ins, outs, sub):
        (pvals,indx) = ins
        (vals,) = outs
        fail = sub['fail']
        return """
        if (CudaNdarray_HOST_DIMS(%(pvals)s)[0] != PyArray_DIMS(%(indx)s)[0])
        {
            PyErr_Format(PyExc_TypeError, "pvals and indx should have same dimensions");
            %(fail)s;
        }
        
        if ((NULL == %(vals)s)
            || (CudaNdarray_HOST_DIMS(%(vals)s)[0] != CudaNdarray_HOST_DIMS(%(pvals)s)[0])
            || (CudaNdarray_HOST_DIMS(%(vals)s)[1] != CudaNdarray_HOST_DIMS(%(pvals)s)[1]))
        {
            Py_XDECREF(%(vals)s);
            
            npy_intp dims[2];
            dims[0] = (CudaNdarray_HOST_DIMS(%(pvals)s)[0]);
            dims[1] = (CudaNdarray_HOST_DIMS(%(pvals)s)[1]);

            %(vals)s = (CudaNdarray*) CudaNdarray_NewDims(2, dims);
            
            if (!%(vals)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc output");
                %(fail)s;
            }
        }
         
        { 
            // NESTED SCOPE
            int nb_rows = CudaNdarray_HOST_DIMS(%(pvals)s)[0];
            int nb_cols = CudaNdarray_HOST_DIMS(%(pvals)s)[1];
            int nb_dims = CudaNdarray_HOST_DIMS(%(pvals)s)[2];
             
            int n_blocks = (nb_rows*nb_cols + 512 - 1) / 512; 

            PyArrayObject *cpu_indices_arr = PyArray_GETCONTIGUOUS(%(indx)s);
     		int* d_indices_arr = (int*)device_malloc(PyArray_NBYTES(cpu_indices_arr));
            if(!d_indices_arr)
                return -1;

     		cudaError_t err = cudaMemcpy(d_indices_arr,
                                         PyArray_DATA(cpu_indices_arr),
                                         PyArray_NBYTES(cpu_indices_arr),
                                         cudaMemcpyHostToDevice);
            
            if(err != cudaSuccess){
                PyErr_Format(
                    PyExc_RuntimeError,
                    "GpuProbsGrabber: cudaMemcpy returned an error: %%s",
                    cudaGetErrorString(err));
                return -1;
            }

            grab_probs_%(name)s<<<n_blocks, 512>>>(
                nb_rows,
                nb_cols,
                nb_dims,
                CudaNdarray_DEV_DATA(%(pvals)s),
                CudaNdarray_DEV_DATA(%(vals)s),
                d_indices_arr
            );
            
            device_free(d_indices_arr);
            Py_XDECREF(cpu_indices_arr);

            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                %(fail)s;
            } 
        } // END NESTED SCOPE
        """ % locals()

    def R_op(self, inputs, eval_points):
        pass

def row_arg_max(var, K):
    return KArgmax(K)(var)

@local_optimizer([KArgmax])
def local_gpu_argmax(node):
    if type(node.op) is KArgmax:
        p, = node.inputs
        vals, indx, = node.outputs
        if (p.dtype == vals.dtype == 'float32' and
            any([i.owner and isinstance(i.owner.op, theano.sandbox.cuda.HostFromGpu) for i in node.inputs])):
            gpu_op = GpuKArgmax(node.op.K)
            ret_vals, ret_indx = gpu_op(gpu_from_host(p))
            return [host_from_gpu(ret_vals), T.cast(host_from_gpu(ret_indx), "int32")]
    if (isinstance(node.op, theano.sandbox.cuda.GpuFromHost) and
        node.inputs[0].owner and type(node.inputs[0].owner.op)
        is KArgmax):
        multi = node.inputs[0].owner
        p, = multi.inputs
        vals, indx, = multi.outputs
        if (p.dtype == vals.dtype == 'float32'):
            gpu_op = GpuKArgmax(node.inputs[0].owner.op.K)
            ret_vals, ret_indx = gpu_op(gpu_from_host(p)) 
            return [gpu_from_host(ret_vals), gpu_from_host(ret_indx)]

@local_optimizer([ProbsGrabber])
def local_probs_grabber(node):
    if type(node.op) is ProbsGrabber:
        p, indx, = node.inputs
        vals, = node.outputs
        if (p.dtype == vals.dtype == 'float32' and
            any([i.owner and isinstance(i.owner.op, theano.sandbox.cuda.HostFromGpu) for i in node.inputs])):
            gpu_op = GpuProbsGrabber()
            ret = gpu_op(gpu_from_host(p),indx)
            return [host_from_gpu(ret),] 
    if (isinstance(node.op, theano.sandbox.cuda.GpuFromHost) and
        node.inputs[0].owner and type(node.inputs[0].owner.op)
        is ProbsGrabber):
        multi = node.inputs[0].owner
        p,indx, = multi.inputs
        vals, = multi.outputs
        if (p.dtype == vals.dtype == 'float32'):
            gpu_op = GpuProbsGrabber()
            ret_vals = gpu_op(gpu_from_host(p),indx) 
            return [gpu_from_host(ret_vals)]

@local_optimizer([Assigner])
def local_assigner(node):
    if type(node.op) is Assigner:
        p, indx, gr, = node.inputs
        vals, = node.outputs
        if (p.dtype == vals.dtype == 'float32' and
            any([i.owner and isinstance(i.owner.op, theano.sandbox.cuda.HostFromGpu) for i in node.inputs])):
            gpu_op = GpuAssigner()
            ret = gpu_op(gpu_from_host(p),indx,gpu_from_host(gr))
            return [host_from_gpu(ret),]
    if (isinstance(node.op, theano.sandbox.cuda.GpuFromHost) and
        node.inputs[0].owner and type(node.inputs[0].owner.op)
        is Assigner):
        multi = node.inputs[0].owner
        p,indx,gr = multi.inputs
        vals, = multi.outputs
        if (p.dtype == vals.dtype == 'float32'):
            gpu_op = GpuAssigner()
            ret_vals = gpu_op(gpu_from_host(p),indx,gpu_from_host(gr)) 
            return [gpu_from_host(ret_vals)]

if cuda_available:
    register_opt()(local_assigner)
    register_opt()(local_probs_grabber)
    register_opt()(local_gpu_argmax)
