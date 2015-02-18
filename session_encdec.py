"""
Session hierarchical encoder-decoder code.
The code is inspired from Encoder Decoder code implemented using Groundhog
library available at => https://github.com/lisa-groundhog/GroundHog/
However, this code does not necessitate the Groundhog library to run.
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Alessandro Sordoni")
__contact__ = "Alessandro Sordoni <sordonia@iro.umontreal>"

import theano
import theano.tensor as T
import numpy as np
import cPickle
import logging

logger = logging.getLogger(__name__)

from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict

from model import *
from utils import *

# Theano speed-up
theano.config.scan.allow_gc = False
#

def add_to_params(params, new_param):
    params.append(new_param)
    return new_param

class EncoderDecoderBase():
    def __init__(self, state, rng, parent):
        self.rng = rng
        self.parent = parent
        self.decoder_bias_all = True
        
        self.state = state
        self.__dict__.update(state)
        
        self.session_rec_activation = eval(self.session_rec_activation)
        self.query_rec_activation = eval(self.query_rec_activation)
         
        self.params = []

class Encoder(EncoderDecoderBase):
    def init_params(self):
        """ Query weights """
        self.W_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.idim, self.rankdim), name='W_emb'))
        self.W_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in'))
        self.W_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='W_hh'))
        self.b_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='b_hh'))
        
        if self.query_step_type == "gated":
            self.W_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in_r'))
            self.W_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in_z'))
            self.W_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='W_hh_r'))
            self.W_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='W_hh_z'))
    
        """ Context weights """
        self.Ws_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.qdim, self.sdim), name='Ws_in'))
        self.Ws_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh'))
        self.bs_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.sdim,), dtype='float32'), name='bs_hh')) 
         
        if self.session_step_type == "gated":
            self.Ws_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.qdim, self.sdim), name='Ws_in_r'))
            self.Ws_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.qdim, self.sdim), name='Ws_in_z'))
            self.Ws_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh_r'))
            self.Ws_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.sdim, self.sdim), name='Ws_hh_z'))

    def plain_query_step(self, x_t, m_t, h_tm1, hr_tm1):
        h_t = self.query_rec_activation(T.dot(x_t, self.W_in) + T.dot(hr_tm1, self.W_hh) + self.b_hh)
         
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
        
        if self.reset_recurrence:
            hr_t = (m_t) * h_t
            # ^ if x_t == </q> then hr_t = 0 (reset for the next step)
            # return both reset state and non-reset state 
        else:
            hr_t = h_t
        return h_t, hr_t
     
    def gated_query_step(self, x_t, m_t, h_tm1, hr_tm1): 
        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_r) + T.dot(hr_tm1, self.W_hh_r))
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_z) + T.dot(hr_tm1, self.W_hh_z))
        h_tilde = self.query_rec_activation(T.dot(x_t, self.W_in) + T.dot(r_t * hr_tm1, self.W_hh) + self.b_hh)
        h_t = (np.float32(1.0) - z_t) * hr_tm1 + z_t * h_tilde
         
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x') 
         
        if self.reset_recurrence:
            hr_t = (m_t) * h_t
            # ^ if x_t == </q> then hr_t = 0 (reset for the next step)
        else:
            hr_t = h_t
        # return both reset state and non-reset state
        return h_t, hr_t, r_t, z_t, h_tilde
    
    def plain_session_step(self, h_t, m_t, hs_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
         
        hs_t = (m_t) * hs_tm1 + (1 - m_t) * self.session_rec_activation(T.dot(h_t, self.Ws_in) + T.dot(hs_tm1, self.Ws_hh) + self.bs_hh)  
        return hs_t

    def gated_session_step(self, h_t, m_t, hs_tm1):
        rs_t = T.nnet.sigmoid(T.dot(h_t, self.Ws_in_r) + T.dot(hs_tm1, self.Ws_hh_r))
        zs_t = T.nnet.sigmoid(T.dot(h_t, self.Ws_in_z) + T.dot(hs_tm1, self.Ws_hh_z))
        hs_tilde = self.session_rec_activation(T.dot(h_t, self.Ws_in) + T.dot(rs_t * hs_tm1, self.Ws_hh) + self.bs_hh)
        hs_t = (np.float32(1.) - zs_t) * hs_tm1 + zs_t * hs_tilde
         
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
         
        hs_t = (m_t) * hs_tm1 + (np.float32(1.) - m_t) * hs_t
        return hs_t, hs_tilde, rs_t, zs_t

    def approx_embedder(self, x):
        return self.W_emb[x]

    def build_encoder(self, x, xmask=None, **kwargs):
        one_step = False
        if len(kwargs):
            one_step = True
         
        # if x.ndim == 2 then 
        # x = (n_steps, batch_size)
        if x.ndim == 2:
            batch_size = x.shape[1]
        # else x = (word_1, word_2, word_3, ...)
        # or x = (last_word_1, last_word_2, last_word_3, ..)
        # in this case batch_size is 
        else:
            batch_size = 1
        
        # if it is not one_step then we initialize everything to 0  
        if not one_step:
            h_0 = T.alloc(np.float32(0), batch_size, self.qdim)
            hr_0 = T.alloc(np.float32(0), batch_size, self.qdim)
            hs_0 = T.alloc(np.float32(0), batch_size, self.sdim) 
        # in sampling mode (i.e. one step) we require 
        else:
            # in this case x.ndim != 2
            assert x.ndim != 2
            assert 'prev_h' in kwargs 
            assert 'prev_hr' in kwargs
            assert 'prev_hs' in kwargs

            h_0 = kwargs['prev_h']
            hr_0 = kwargs['prev_hr']
            hs_0 = kwargs['prev_hs']

        xe = self.approx_embedder(x)
        if not xmask:
            xmask = T.neq(x, self.eoq_sym)

        # Gated Encoder
        if self.query_step_type == "gated":
            f_enc = self.gated_query_step
            o_enc_info = [h_0, hr_0, None, None, None]
        else:
            f_enc = self.plain_query_step
            o_enc_info = [h_0, hr_0]

        if self.session_step_type == "gated":
            f_hier = self.gated_session_step
            o_hier_info = [hs_0, None, None, None]
        else:
            f_hier = self.plain_session_step
            o_hier_info = [hs_0]
        
        # Run through all the sentence (encode everything)
        if not one_step: 
            _res, _ = theano.scan(f_enc,
                              sequences=[xe, xmask],\
                              outputs_info=o_enc_info) 
        # Make just one step further
        else:
            _res = f_enc(xe, xmask, h_0, hr_0)

        h = _res[0]
        hr = _res[1]
        
        # All hierarchical sentence
        if not one_step:
            _res,  _ = theano.scan(f_hier,\
                               sequences=[h, xmask],\
                               outputs_info=o_hier_info)
        # Just one step further
        else:
            _res = f_hier(h, xmask, hs_0)

        if isinstance(_res, list) or isinstance(_res, tuple):
            hs = _res[0]
        else:
            hs = _res

        return h, hr, hs 

    def __init__(self, state, rng, parent):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.init_params()

class Decoder(EncoderDecoderBase):
    
    EVALUATION = 0
    SAMPLING = 1
    BEAM_SEARCH = 2

    def __init__(self, state, rng, parent, encoder):
        EncoderDecoderBase.__init__(self, state, rng, parent)
        self.trng = RandomStreams(self.seed)
        self.encoder = encoder
        self.init_params()

    def init_params(self): 
        """ Decoder weights """
        self.bd_out = add_to_params(self.params, theano.shared(value=np.zeros((self.idim,), dtype='float32'), name='bd_out'))
        self.Wd_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.idim), name='Wd_emb'))
        self.Wd_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='Wd_hh'))
        self.bd_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='bd_hh'))
        self.Wd_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='Wd_in')) 
        self.Wd_s_q = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.sdim, self.qdim), name='Wd_s_q'))
        
        if self.query_step_type == "gated":
            self.Wd_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='Wd_in_r'))
            self.Wd_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='Wd_in_z'))
            self.Wd_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='Wd_hh_r'))
            self.Wd_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='Wd_hh_z'))
            
            if self.decoder_bias_all:
                self.Wd_s_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.sdim, self.qdim), name='Wd_s_z'))
                self.Wd_s_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.sdim, self.qdim), name='Wd_s_r'))
        
        """ Rank """
        if hasattr(self, 'train_rank'):
            self.Wr_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.sdim, 1), name='Wr_out'))
            self.br_out = add_to_params(self.params, theano.shared(value=np.zeros((1,), dtype='float32'), name='br_out'))
        
        ######################   
        # Output layer weights
        ######################
        out_target_dim = self.qdim
        if not self.maxout_out:
            out_target_dim = self.rankdim

        self.Wd_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.qdim, out_target_dim), name='Wd_out'))
         
        # Set up deep output
        if self.deep_out:
            self.Wd_e_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, out_target_dim), name='Wd_e_out'))
            self.bd_e_out = add_to_params(self.params, theano.shared(value=np.zeros((out_target_dim,), dtype='float32'), name='bd_e_out'))
             
            if self.decoder_bias_all:
                self.Wd_s_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.sdim, out_target_dim), name='Wd_s_out'))
   
    def build_rank_layer(self, hs):
        return T.dot(hs, self.Wr_out) + self.br_out

    def build_output_layer(self, hs, xd, hd):
        pre_activ = T.dot(hd, self.Wd_out)

        if self.deep_out:
            pre_activ += T.dot(xd, self.Wd_e_out) + self.bd_e_out
            
            if self.decoder_bias_all:
                pre_activ += T.dot(hs, self.Wd_s_out)
                # ^ if bias all, bias the deep output
         
        if self.maxout_out:
            pre_activ = Maxout(2)(pre_activ)
         
        return SoftMax(T.dot(pre_activ, self.Wd_emb) + self.bd_out)
        
    def approx_embedder(self, x):
        return self.encoder.approx_embedder(x)

    def build_next_probs_predictor(self, hs, x, prev_hd):
        """ 
        Return output probabilities given prev_words x, hierarchical pass hs, and previous hd
        hs should always be the same (and should not be updated).
        """
        return self.build_decoder(hs, x, mode=Decoder.BEAM_SEARCH, prev_hd=prev_hd)

    def build_decoder(self, hs, x, xmask=None, y=None, mode=EVALUATION, prev_hd=None, step_num=None):
        # Check parameter consistency
        if mode == Decoder.EVALUATION:
            assert not prev_hd
            assert y
        else:
            assert not y
            assert prev_hd
         
        # if mode == EVALUATION
        #   xd = (timesteps, batch_size, qdim)
        #
        # if mode != EVALUATION
        #   xd = (n_samples, dim)
        xd = self.approx_embedder(x)
        if not xmask:
            xmask = T.neq(x, self.eoq_sym) 
        
        # we must zero out the </q> embedding
        if xd.ndim != 3:
            assert mode != Decoder.EVALUATION
            xd = (xd.dimshuffle((1, 0)) * xmask).dimshuffle((1, 0))
        else:
            assert mode == Decoder.EVALUATION
            xd = (xd.dimshuffle((2,0,1)) * xmask).dimshuffle((1,2,0))
        
        # Run the decoder
        if mode == Decoder.EVALUATION:
            hd_init = T.alloc(np.float32(0), x.shape[1], self.qdim)
        else:
            hd_init = prev_hd 

        if self.query_step_type == "gated":
            f_dec = self.gated_step
            o_dec_info = [hd_init, None, None, None] 
        else:
            f_dec = self.plain_step
            o_dec_info = hd_init
        
        # If the mode of the decoder is EVALUATION
        # then we evaluate by default all the sentence
        # xd - i.e. xd.ndim == 3, xd = (timesteps, batch_size, qdim)
        if mode == Decoder.EVALUATION: 
            _res, _ = theano.scan(f_dec,
                              sequences=[xd, xmask, hs],\
                              outputs_info=o_dec_info)
        # else we evaluate only one step of the recurrence using the
        # previous hidden states and the previous computed hierarchical 
        # states.
        else:
            _res = f_dec(xd, xmask, hs, prev_hd)

        if isinstance(_res, list) or isinstance(_res, tuple):
            hd = _res[0]
        else:
            hd = _res
        
        outputs = self.build_output_layer(hs, xd, hd)
        ranks = self.build_rank_layer(hs)
         
        # EVALUATION  : Return target_probs + all the predicted ranks
        # target_probs.ndim == 3, ranks.ndim == 3
        if mode == Decoder.EVALUATION:
            target_probs = GrabProbs(outputs, y)
            return target_probs, ranks 
        # BEAM_SEARCH : Return output (the softmax layer) + ranks + the new hidden states
        elif mode == Decoder.BEAM_SEARCH:
            return outputs, hd
        # SAMPLING    : Return a vector of n_sample from the output layer 
        #                 + log probabilities + the new hidden states
        elif mode == Decoder.SAMPLING:
            if outputs.ndim == 1:
                outputs = outputs.dimshuffle('x', 0)
             
            sample = self.trng.multinomial(pvals=outputs, dtype='int64').argmax(axis=-1)
            if outputs.ndim == 1:
                sample = sample[0]
             
            log_prob = -T.log(T.diag(outputs.T[sample]))
            return sample, log_prob, hd
    
    def sampling_step(self, *args): 
        args = iter(args)

        # Arguments that correspond to scan's "sequences" parameteter:
        step_num = next(args)
        assert step_num.ndim == 0
        
        # Arguments that correspond to scan's "outputs" parameteter:
        prev_word = next(args)
        assert prev_word.ndim == 1
        
        # skip the previous word log probability
        log_prob = next(args)
        assert log_prob.ndim == 1
        
        prev_h = next(args) 
        assert prev_h.ndim == 2
        
        prev_hr = next(args)
        assert prev_hr.ndim == 2
        
        prev_hs = next(args)
        assert prev_hs.ndim == 2
        
        prev_hd = next(args)
        assert prev_hd.ndim == 2
       
        # When we sample we shall recompute the encoder for one step...
        encoder_args = dict(prev_hr=prev_hr, prev_hs=prev_hs, prev_h=prev_h)
        h, hr, hs = self.parent.encoder.build_encoder(prev_word, **encoder_args)
         
        assert h.ndim == 2
        assert hr.ndim == 2
        assert hs.ndim == 2
        
        # ...and decode one step.
        sample, log_prob, hd = self.build_decoder(hs, prev_word, prev_hd=prev_hd, step_num=step_num, mode=Decoder.SAMPLING)
        
        assert sample.ndim == 1
        assert log_prob.ndim == 1
        assert hd.ndim == 2

        return [sample, log_prob, h, hr, hs, hd]
    
    def build_sampler(self, n_samples, n_steps):
        # For the naive sampler, the states are:
        # 1) a vector [</q>] * n_samples to seed the sampling
        # 2) a vector of [ 0. ] * n_samples for the log_probs
        # 3) prev_h hidden layers
        # 4) prev_hr hidden layers
        # 5) prev_hs hidden layers
        # 6) prev_hd hidden layers
        states = [T.alloc(np.int64(self.eoq_sym), n_samples),
                  T.alloc(np.float32(0.), n_samples),
                  T.alloc(np.float32(0.), n_samples, self.qdim),
                  T.alloc(np.float32(0.), n_samples, self.qdim),
                  T.alloc(np.float32(0.), n_samples, self.sdim),
                  T.alloc(np.float32(0.), n_samples, self.qdim)]
        outputs, updates = theano.scan(self.sampling_step,
                    outputs_info=states,
                    sequences=[T.arange(n_steps, dtype='int64')], 
                    n_steps=n_steps,
                    name="sampler_scan")
        # Return sample, log_probs and updates (for tnrg multinomial)
        return (outputs[0], outputs[1]), updates

    def gated_step(self, xd_t, m_t, hs_t, hd_tm1): 
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
         
        if self.reset_recurrence:
            # ^ iff xd_t = </q> (m_t = 0) then xd_t = 0 
            hd_tm1 = (m_t) * hd_tm1
         
        if not self.decoder_bias_all:
            # Do not bias all the decoder (force to store very useful information in the first state)
            hd_tm1 += (1 - m_t) * T.tanh(T.dot(hs_t, self.Wd_s_q) + self.bd_hh)
            # ^ iff xd_t = </q> (m_t = 0) then hd_tm1n = init_state
            rd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_r) + T.dot(hd_tm1, self.Wd_hh_r))
            zd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_z) + T.dot(hd_tm1, self.Wd_hh_z))
            hd_tilde = self.query_rec_activation(T.dot(xd_t, self.Wd_in) \
                                        + T.dot(rd_t * hd_tm1, self.Wd_hh) \
                                        + self.bd_hh)
        else:
            rd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_r) + T.dot(hd_tm1, self.Wd_hh_r) + T.dot(hs_t, self.Wd_s_r))
            zd_t = T.nnet.sigmoid(T.dot(xd_t, self.Wd_in_z) + T.dot(hd_tm1, self.Wd_hh_z) + T.dot(hs_t, self.Wd_s_z))
            hd_tilde = self.query_rec_activation(T.dot(xd_t, self.Wd_in) \
                                        + T.dot(rd_t * hd_tm1, self.Wd_hh) \
                                        + T.dot(hs_t, self.Wd_s_q) \
                                        + self.bd_hh)
        hd_t = (np.float32(1.) - zd_t) * hd_tm1 + zd_t * hd_tilde 
        return hd_t, rd_t, zd_t, hd_tilde
    
    def plain_step(self, xd_t, m_t, hs_t, hd_tm1):
        if m_t.ndim >= 1:
            m_t = m_t.dimshuffle(0, 'x')
        
        if self.reset_recurrence:
            # We already assume that xd are zeroed out
            # ^ iff xd_t = </q> (m_t = 0) then xd_t = 0 
            hd_tm1 = (m_t) * hd_tm1
            # ^ iff xd_t = </q> (m_t = 0) then hd_tm1n = init_state
        
        if not self.decoder_bias_all:
            # Do not bias all the decoder (force to store very useful information in the first state) 
            hd_tm1 += (1 - m_t) * T.tanh(T.dot(hs_t, self.Wd_s_q) + self.bd_hh)
            hd_t = self.query_rec_activation( T.dot(xd_t, self.Wd_in) \
                                             + T.dot(hd_tm1, self.Wd_hh) \
                                             + self.bd_hh )
        else:
            hd_t = self.query_rec_activation( T.dot(xd_t, self.Wd_in) \
                                             + T.dot(hd_tm1, self.Wd_hh) \
                                             + T.dot(hs_t, self.Wd_s_q) \
                                             + self.bd_hh )
        return hd_t
    ####

class SessionEncoderDecoder(Model):
    def indices_to_words(self, seq):
        sen = []
        ses = []
        for k in range(len(seq)):
            if seq[k] == self.eos_sym:
                break
            if seq[k] == self.soq_sym:
                continue
            if seq[k] == self.eoq_sym:
                ses.append(" ".join(sen))
                sen = []
            else:
                sen.append(self.idx_to_str[seq[k]])
        return ses

    def words_to_indices(self, seq, add_se=True):
        sen = []
        for k in range(len(seq)):
            sen.append(self.str_to_idx.get(seq[k], self.unk_sym))

        if add_se and len(sen) > 0 \
           and (sen[0] != self.soq_sym or sen[-1] != self.eoq_sym):
            if sen[0] != self.soq_sym:
                sen = [self.soq_sym] + sen
            if sen[-1] != self.eoq_sym:
                sen = sen + [self.eoq_sym]
        return sen

    def compute_updates(self, training_cost, params):
        updates = []
        
        grads = T.grad(training_cost, params)
        grads = OrderedDict(zip(params, grads))
        
        # Clip stuff
        c = numpy.float32(self.cutoff)
        clip_grads = []
        
        norm_gs = T.sqrt(sum(T.sum(g ** 2) for p, g in grads.items()))
        normalization = T.switch(T.ge(norm_gs, c), c / norm_gs, np.float32(1.))
        notfinite = T.or_(T.isnan(norm_gs), T.isinf(norm_gs))
         
        for p, g in grads.items():
            # tmpg = T.switch(T.ge(norm_gs, c), g*c/norm_gs, g)
            clip_grads.append((p, T.switch(notfinite, numpy.float32(.1) * p, g * normalization)))
        grads = OrderedDict(clip_grads)

        if self.updater == 'adagrad':
            updates = Adagrad(grads, self.lr)  
        elif self.updater == 'sgd':
            raise Exception("Sgd not implemented!")
        elif self.updater == 'adadelta':
            updates = Adadelta(grads)
        elif self.updater == 'rmsprop':
            updates = RMSProp(grads, self.lr)
        elif self.updater == 'adam':
            updates = Adam(grads)
        else:
            raise Exception("Updater not understood!") 
        return updates
   
    def build_train_function(self):
        if not hasattr(self, 'train_fn'):
            # Compile functions
            logger.debug("Building train function") 
            self.train_fn = theano.function(inputs=[self.x_data, self.x_ranks, self.x_max_length, self.x_cost_mask],
                                            outputs=self.training_cost,
                                            updates=self.updates, name="train_fn") 
        return self.train_fn

    def build_eval_function(self):
        if not hasattr(self, 'eval_fn'):
            # Compile functions
            logger.debug("Building evaluation function")
            self.eval_fn = theano.function(inputs=[self.x_data, self.x_ranks, self.x_max_length, self.x_cost_mask], 
                                            outputs=[self.prediction_cost, self.rank_cost, self.training_cost], name="eval_fn")
        return self.eval_fn

    def build_sampling_function(self):
        if not hasattr(self, 'sample_fn'):
            logger.debug("Building sampling function")
            self.sample_fn = theano.function(inputs=[self.n_samples, self.n_steps], outputs=[self.sample, self.sample_log_prob], \
                                       updates=self.sampling_updates, name="sample_fn")
        return self.sample_fn

    def build_rank_prediction_function(self):
        if not hasattr(self, 'rank_fn'):
            h, hr, hs = self.encoder.build_encoder(self.aug_x_data)
            ranks = self.decoder.build_rank_layer(hs)
            self.rank_fn = theano.function(
                inputs=[self.x_data],
                outputs=[ranks],
                name="rank_fn")
        return self.rank_fn

    def build_next_probs_function(self):
        if not hasattr(self, 'next_probs_fn'):
            outputs, hd = self.decoder.build_next_probs_predictor(self.beam_hs, self.beam_source, prev_hd=self.beam_hd)
            self.next_probs_fn = theano.function(
                inputs=[self.beam_hs, self.beam_source, self.beam_hd],
                outputs=[outputs, hd],
                name="next_probs_fn")
        return self.next_probs_fn

    def build_encoder_function(self):
        if not hasattr(self, 'encoder_fn'):
            h, hr, hs = self.encoder.build_encoder(self.aug_x_data)
            self.encoder_fn = theano.function(
                inputs=[self.x_data],
                outputs=[h, hr, hs],
                name="encoder_fn")
        return self.encoder_fn

    def __init__(self, state):
        Model.__init__(self)    
        self.state = state
        
        # Compatibility towards older models
        self.decoder_bias_all = True
        self.__dict__.update(state)
        
        self.rng = np.random.RandomState(self.seed) 

        # Load dictionary 
        self.str_to_idx = cPickle.load(open(self.dictionary, 'r'))
        self.idx_to_str = dict([(tok_id, tok) for tok, tok_id in self.str_to_idx.items()])

        if '</q>' not in self.str_to_idx \
           or '</s>' not in self.str_to_idx:
                raise Exception("Error, malformed dictionary!")
         
        # Number of words in the dictionary 
        self.idim = len(self.str_to_idx)
        self.state['idim'] = self.idim
 
        logger.debug("Initializing encoder")
        self.encoder = Encoder(self.state, self.rng, self)
        logger.debug("Initializing decoder")
        self.decoder = Decoder(self.state, self.rng, self, self.encoder)
        
        # Init params
        self.params = self.encoder.params + self.decoder.params
        assert len(set(self.params)) == (len(self.encoder.params) + len(self.decoder.params))
        
        self.x_data = T.imatrix('x_data')
        self.x_ranks = T.matrix('x_ranks')
        self.x_cost_mask = T.matrix('cost_mask')
        self.x_max_length = T.iscalar('x_max_length')

        # The training is done with a trick. We append a special </q> at the beginning of the session
        # so that we can predict also the first query in the session starting from the session beginning token (</q>).
        self.aug_x_data = T.concatenate([T.alloc(np.int32(self.eoq_sym), 1, self.x_data.shape[1]), self.x_data]) 
        training_x = self.aug_x_data[:self.x_max_length]
        training_y = self.aug_x_data[1:self.x_max_length+1]
        training_ranks = self.x_ranks[:self.x_max_length-1].flatten()
        
        training_hs_mask = T.neq(training_x, self.eoq_sym)
        training_x_cost_mask = self.x_cost_mask[:self.x_max_length].flatten()
        # training_x_ranks_mask = training_hs_mask[:].flatten()
        training_x_ranks_mask = T.neq(training_ranks, 0).flatten()
        # ^ The training cost mask is the hs_mask shifted one position to the right 
         
        h, hr, hs = self.encoder.build_encoder(training_x, xmask=training_hs_mask)
        target_probs, target_ranks = self.decoder \
                .build_decoder(hs, training_x, xmask=training_hs_mask, y=training_y, mode=Decoder.EVALUATION)
        target_ranks = target_ranks[1:].flatten()
        # ^ skip first target rank 
        
        # Prediction cost and rank cost
        self.prediction_cost = T.sum(-T.log(target_probs) * training_x_cost_mask)
         
        # Mean squared error
        # sum_queries = T.sum(training_x_ranks_mask) # Number of queries in the batch
        self.rank_cost = T.sum(((target_ranks - training_ranks) ** 2) * (training_x_ranks_mask))

        self.training_cost = self.prediction_cost + np.float32(self.lambda_rank) * self.rank_cost
        self.updates = self.compute_updates(self.training_cost / training_x.shape[1], self.params)
        
        # Sampling variables
        self.n_samples = T.iscalar("n_samples")
        self.n_steps = T.iscalar("n_steps")
        (self.sample, self.sample_log_prob), self.sampling_updates \
                = self.decoder.build_sampler(self.n_samples, self.n_steps) 

        # Beam-search variables
        self.beam_source = T.lvector("beam_source")
        self.beam_hs = T.matrix("beam_hs")
        self.beam_step_num = T.lscalar("beam_step_num")
        self.beam_hd = T.matrix("beam_hd")
