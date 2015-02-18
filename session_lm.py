import theano
import theano.tensor as T
import numpy as np
import cPickle
import logging
logger = logging.getLogger(__name__)

from model import *
from utils import *

from theano.sandbox.scan import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nnet.conv3d2d import *

class SessionLM(Model):
    def InitParams(self):
        self.params = []
        self.W_hh = theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='W_hh')
        self.W_emb = theano.shared(value=NormalInit(self.rng, self.idim, self.rankdim), name='W_emb')
        self.W_in = theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in')
        self.b_hh = theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='b_hh')
        
        # Deep output 
        self.W_out = theano.shared(value=NormalInit(self.rng, self.qdim, self.qdim), name='W_out')
        self.b_out = theano.shared(value=np.zeros((self.idim,), dtype='float32'), name='b_out') 
        self.W_e_out = theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_e_out')
        self.W_s_out = theano.shared(value=NormalInit(self.rng, self.sdim, self.qdim), name='W_s_out')        
        self.b_e_out = theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='b_e_out')
        self.params += [self.W_e_out, self.W_s_out, self.b_e_out, self.W_out, \
                        self.W_hh, self.W_emb, self.W_in, self.b_out, self.b_hh]

        """ Gated variant gates only the upper level rnn """
        if self.query_step_type == "gated":
            self.W_in_r = theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in_r')
            self.W_in_z = theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in_z')
            self.W_hh_r = theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='W_hh_r')
            self.W_hh_z = theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='W_hh_z')
            self.W_s_z = theano.shared(value=NormalInit(self.rng, self.sdim, self.qdim), name='W_s_z')
            self.W_s_r = theano.shared(value=NormalInit(self.rng, self.sdim, self.qdim), name='W_s_r')
            self.params += [self.W_in_r, self.W_in_z, self.W_hh_r, self.W_hh_z, self.W_s_z, self.W_s_r] 
 
        self.Ws_in = theano.shared(value=NormalInit(self.rng, self.qdim, self.sdim), name='Ws_in')
        self.Ws_hh = theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh')
        self.W_s_q = theano.shared(value=NormalInit(self.rng, self.sdim, self.qdim), name='W_s_q')
        self.bs_hh = theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_hh')
        self.params += [self.Ws_in, self.Ws_hh, self.W_s_q, self.bs_hh]
 
        # Rank prediction
        self.Wr_out = theano.shared(value=NormalInit(self.rng, self.sdim, 1), name='Wr_out')
        self.br_out = theano.shared(value=np.zeros((1,), dtype='float32'), name='br_out')
        self.params += [self.Wr_out, self.br_out]

        if self.session_step_type == "gated":
            self.Ws_in_r = theano.shared(value=NormalInit(self.rng, self.qdim, self.sdim), name='Ws_in_r')
            self.Ws_in_z = theano.shared(value=NormalInit(self.rng, self.qdim, self.sdim), name='Ws_in_z') 
            self.Ws_hh_r = theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh_r')
            self.Ws_hh_z = theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh_z')
            self.params += [ self.Ws_in_r, self.Ws_in_z, self.Ws_hh_r, self.Ws_hh_z] 
        
        self.l2 = {}
        for param in self.params:
            self.l2[param] = T.sqrt((param ** 2).sum())

    def Sample(self, length=20):
        samples = numpy.zeros((length, 1)).astype('int32')
        samples[0] = [self.soq_sym]
        step = 1
        ranks = []

        while True: 
            self.gpuData.set_value(samples)
            self.symMaxStep.set_value(np.int32(len(samples)))
            
            x_step, rank = self.SymSample(step)
            samples[step] = x_step
            
            if samples[step - 1] == self.eoq_sym:
                ranks.append("%.4f" % rank.flatten()[0])
                # step += 1
                # samples[step] = self.soq_sym

            step += 1
            if step >= length-1 or x_step == [self.eos_sym]:
                break
        
        sampled = []
        for sample in samples.ravel(): 
            if sample == self.soq_sym:
                sampled.append("[")
            elif sample == self.eoq_sym:
                sampled.append("]")
            elif sample == self.eos_sym:
                break
            else:
                sampled.append(self.id2string[sample])
        return " ".join(sampled) + " - " + ",".join(ranks)

    def Step(self, x_t, m_t, h_tm1, hr_tm1, hs_tm1):
        """ Gated H-RNN forward step """ 
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
        
        if self.query_step_type == "gated":
            r_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_r) + T.dot(hr_tm1, self.W_hh_r) + T.dot(hs_tm1, self.W_s_r))
            z_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_z) + T.dot(hr_tm1, self.W_hh_z) + T.dot(hs_tm1, self.W_s_z))
            h_tilde = self.query_rec_activation(T.dot(x_t, self.W_in) \
                                        + T.dot(r_t * hr_tm1, self.W_hh) \
                                        + T.dot(hs_tm1, self.W_s_q) \
                                        + self.b_hh)
            h_t = (np.float32(1.) - z_t) * hr_tm1 + z_t * h_tilde 
        else:
            h_t = self.query_rec_activation(T.dot(x_t, self.W_in) \
                                            + T.dot(hr_tm1, self.W_hh) \
                                            + T.dot(hs_tm1, self.W_s_q) \
                                            + self.b_hh) 
         
        if self.session_step_type == "gated":
            rs_t = T.nnet.sigmoid(T.dot(h_t, self.Ws_in_r) + T.dot(hs_tm1, self.Ws_hh_r))
            zs_t = T.nnet.sigmoid(T.dot(h_t, self.Ws_in_z) + T.dot(hs_tm1, self.Ws_hh_z))
            hs_tilde = self.session_rec_activation(T.dot(h_t, self.Ws_in) + T.dot(rs_t * hs_tm1, self.Ws_hh) + self.bs_hh)
            hs_u = (np.float32(1.) - zs_t) * hs_tm1 + zs_t * hs_tilde 
        else:
            hs_u = self.session_rec_activation(T.dot(h_t, self.Ws_in) + T.dot(hs_tm1, self.Ws_hh) + self.bs_hh)
         
        hs_t = (m_t) * hs_tm1 + (np.float32(1.0) - m_t) * hs_u 

        if self.reset_recurrence:
            hr_t = m_t * h_t
        else:
            hr_t = h_t
        return h_t, hr_t, hs_t
    
    def Out(self, x, h, hs):
        # Word prediction
        pre_activ = T.dot(x, self.W_e_out) + T.dot(h, self.W_out) + T.dot(hs, self.W_s_out) + self.b_e_out
        maxout = Maxout(2)(pre_activ)
        
        # Rank prediction
        rank = T.dot(hs, self.Wr_out) + self.br_out
        return SoftMax(T.dot(maxout, self.W_emb.T) + self.b_out), rank
     
    def FullCost(self, output, isTraining=True):
        word_probs, rank = output
        cost = T.sum(-T.log(GrabProbs(word_probs, self.symY)) * self.symCostMap.flatten())
        cost += T.sum(((rank.flatten() - self.symRank.flatten()) ** 2) * (1. - self.symMap.flatten()))
        return cost

    def Valid(self, minibatch):
        # Upload minibatch on GPU    
        self.gpuCostMap.set_value(minibatch['x_mask'].astype('int32'))
        self.gpuData.set_value(minibatch['x'].astype('int32'))
        self.gpuRankData.set_value(minibatch['y'].astype('int32'))
        self.symMaxStep.set_value(numpy.int32(minibatch['max_length'])) 
        return self.SymValid()
    
    def Train(self, minibatch):
        # Upload minibatch on GPU    
        self.gpuCostMap.set_value(minibatch['x_mask'].astype('int32'))
        self.gpuData.set_value(minibatch['x'].astype('int32'))
        self.gpuRankData.set_value(minibatch['y'].astype('int32'))
        self.symMaxStep.set_value(numpy.int32(minibatch['max_length']))
        return self.SymTrainFull()

    def ComputeUpdates(self, symOptCost):
        updates = []
        grads = dict([ (param, T.grad(symOptCost, param, disconnected_inputs='warn')) for param in self.params]) 
        
        # Clip stuff
        c = numpy.float32(self.cutoff)
        
        norm_gs = T.sqrt(sum(T.sum(g ** 2) for p, g in grads.items()))
        notfinite = T.or_(T.isnan(norm_gs), T.isinf(norm_gs))
        clip_grads = []
        for p, g in grads.items():
            norm = T.sqrt(T.sum(g**2))
            tmpg = T.switch(T.ge(norm_gs, c), g*c/norm_gs, g)
            clip_grads.append((p, T.switch(notfinite, numpy.float32(.1)*p, tmpg)))
        grads = OrderedDict(clip_grads)
        
        if self.updater == 'adagrad':
            updates = Adagrad(grads, self.lr)  
        elif self.updater == 'sgd':
            raise Exception("Sgd not implemented!")
        elif self.updater == 'adadelta':
            updates = Adadelta(grads)
        elif self.updater == 'rmsprop':
            updates = RMSProp(grads, self.lr)
        else:
            raise Exception("Updater not understood!") 
        return updates

    def __init__(self, state):
        Model.__init__(self)

        self.state = state
        self.__dict__.update(state)
        self.rng = np.random.RandomState(1) 
        self.trng = RandomStreams(1)

        # Load dictionary 
        self.string2id = cPickle.load(open(self.dictionary, 'r'))
        self.id2string = dict([(idtok, tok) for tok, idtok in self.string2id.items()])

        if '<q>' not in self.string2id \
           or '</q>' not in self.string2id \
           or '</s>' not in self.string2id:
                raise Exception("Error, malformed dictionary!")
        
        # Number of words in the dictionary 
        self.idim = len(self.string2id) 
        self.session_rec_activation = eval(self.session_rec_activation)
        self.query_rec_activation = eval(self.query_rec_activation)

        # Init params
        self.InitParams()
         
        # Input structures
        self.gpuData = theano.shared(value=np.zeros((self.seqlen, self.bs)).astype('int32'), name='gpuData')
        self.gpuRankData = theano.shared(value=np.zeros((self.seqlen, self.bs)).astype('int32'), name='gpuRankData')

        self.gpuCostMap = theano.shared(value=np.zeros((self.seqlen, self.bs)).astype('int32'), name='gpuCostMap') 
        self.symMaxStep = theano.shared(value=np.int32(0), name='maxStep')

        self.symX = self.gpuData[:self.symMaxStep-1, :]
        self.symRank = self.gpuRankData[:self.symMaxStep-1, :]
        self.symY = self.gpuData[1:self.symMaxStep, :]
        
        self.symMap = T.neq(self.symX, self.eoq_sym)
        
        # <q> a  b  c   </q>  <q>
        # a   b  c </q>  <q>  </s>
        self.symCostMap = self.gpuCostMap[1:self.symMaxStep, :]
        
        logger.debug("Compiling train function")
        logger.debug("Reset recurrence is set to {}".format(self.reset_recurrence))

        f = self.Step
        xi = self.W_emb[self.symX]
        sequences = [xi, self.symMap]
        outputs_info = [T.alloc(np.float32(0), self.symX.shape[1], self.qdim),\
                        T.alloc(np.float32(0), self.symX.shape[1], self.qdim),\
                        T.alloc(np.float32(0), self.symX.shape[1], self.sdim)]
        
        (h, hr, hs), _ = theano.scan(f, sequences=sequences, outputs_info=outputs_info)
         
        o = self.Out(xi, h, hs)
        full_cost = self.FullCost(o)
        
        updates = self.ComputeUpdates(full_cost/self.symX.shape[1])
        
        self.SymTrainFull = theano.function(inputs=[], outputs=full_cost, updates=updates)
        self.SymValid = theano.function(inputs=[], outputs=full_cost)
        
        """ Sampling Function """
        symSampleStep = T.iscalar()
        output = self.Out(xi[symSampleStep-1], h[symSampleStep-1], hs[symSampleStep-1])
        word_probs, rank = output
        x_step = self.trng.multinomial(pvals=word_probs, dtype='int64').argmax(axis=-1).flatten() 
        self.SymSample = theano.function(inputs=[symSampleStep], outputs=[x_step, rank])
