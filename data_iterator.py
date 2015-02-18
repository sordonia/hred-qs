__docformat__ = 'restructedtext en'
__authors__ = ("Alessandro Sordoni")
__contact__ = "Alessandro Sordoni <sordonia@iro.umontreal>"

import numpy as np
import theano
import theano.tensor as T
import sys, getopt
import logging

from state import *
from utils import *
from SS_dataset import *

import itertools
import sys
import pickle
import random
import datetime

logger = logging.getLogger(__name__)

def create_padded_batch(state, x, y=None):
    mx = state['seqlen']
    n = state['bs'] 
    
    X = numpy.zeros((mx, n), dtype='int32')
    Y = numpy.zeros((mx, n), dtype='float32')
    Xmask = numpy.zeros((mx, n), dtype='float32') 

    # Fill X and Xmask
    # Keep track of number of predictions and maximum sentence length
    num_preds = 0
    max_length = 0
    
    for idx in xrange(len(x[0])):
        # Insert sequence idx in a column of matrix X
        session_length = len(x[0][idx])

        # Fiddle-it if it is too long ..
        eoq_idx = numpy.where(numpy.array(x[0][idx][:mx - 2]) == state['eoq_sym'])[0]
        if mx < session_length: 
            # .. but if the first query is longer than the session
            # we cannot fix it, so we skip it
            if not len(eoq_idx):
                continue

            # fix it
            x[0][idx][eoq_idx[-1] + 1] = state['eos_sym']
            assert eoq_idx[-1] + 2 < mx
            session_length = eoq_idx[-1] + 2
         
        X[:session_length, idx] = x[0][idx][:session_length]
        
        if y:
            Y[eoq_idx, idx] = y[0][idx][:len(eoq_idx)]

        max_length = max(max_length, session_length)
        # Set the number of predictions == sum(Xmask), for cost purposes
        num_preds += session_length
        
        # Mark the end of phrase
        if len(x[0][idx]) < mx:
            X[session_length:, idx] = state['eos_sym']

        # Initialize Xmask column with ones in all positions that
        # were just set in X
        Xmask[:session_length, idx] = 1.
     
    assert num_preds == numpy.sum(Xmask)
    return {'x': X, 'y': Y, 'x_mask': Xmask, 'num_preds': num_preds, 'max_length': max_length}

def get_batch_iterator(state):
    class Iterator(SSIterator):
        def __init__(self, *args, **kwargs):
            SSIterator.__init__(self, *args, **kwargs)
            self.batch_iter = None
    
        def get_homogenous_batch_iter(self):
            while True:
                k_batches = state['sort_k_batches']
                batch_size = state['bs']
               
                data = []
                for k in range(k_batches):
                    batch = SSIterator.next(self)
                    if batch:
                        data.append(batch)
                
                if not len(data):
                    return
                
                sessions = data
                if self.has_ranks:
                    sessions, ranks = zip(*data)
                    y = numpy.asarray(list(itertools.chain(*ranks)))

                x = numpy.asarray(list(itertools.chain(*sessions)))
                lens = numpy.asarray([map(len, x)])
                order = numpy.argsort(lens.max(axis=0)) if state['sort_k_batches'] > 1 \
                        else numpy.arange(len(x))
                
                for k in range(len(data)):
                    indices = order[k * batch_size:(k + 1) * batch_size]
                    if self.has_ranks:
                        batch = create_padded_batch(state, [x[indices]], [y[indices]])
                    else:
                        batch = create_padded_batch(state, [x[indices]])
                    if batch:
                        yield batch
        
        def start(self):
            SSIterator.start(self)
            self.batch_iter = None

        def next(self):
            if not self.batch_iter:
                self.batch_iter = self.get_homogenous_batch_iter()
            try:
                batch = next(self.batch_iter)
            except StopIteration:
                return None
            return batch

    train_data = Iterator(
        batch_size=int(state['bs']),
        session_file=state['train_session'],
        rank_file=state.get('train_rank', None),
        queue_size=100,
        use_infinite_loop=True,
        max_len=state['seqlen']) 
     
    valid_data = Iterator(
        batch_size=int(state['bs']),
        session_file=state['valid_session'],
        rank_file=state.get('valid_rank', None),
        use_infinite_loop=False,
        queue_size=100,
        max_len=state['seqlen'])
    
    return train_data, valid_data

if __name__ == '__main__':
    state = prototype_rfp()
    train_data, valid_data, test_data =  get_batch_iterator(state)
    train_data.start()

    cpt = 0
    while True:
        print "Done ", cpt
        batch = train_data.next()
        if not batch:
            print "Restart"
            continue
        cpt += 1
        # print batch['max_length']
        # print batch['x'][:,0]
        # print batch['x'][:,1]
        print batch['x'].shape
        print batch['y'].shape
        # print batch['y'][:,1]
        # print batch['y_mask']
        break
