#!/usr/bin/env python
__docformat__ = 'restructedtext en'
__authors__ = ("Alessandro Sordoni")
__contact__ = "Alessandro Sordoni <sordonia@iro.umontreal>"

from data_iterator import *
from state import *
from session_encdec import *
from session_lm import *
from utils import *

import time
import traceback
import os.path
import sys
import argparse
import cPickle
import logging
import pprint
import numpy
import collections
import signal

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
logger = logging.getLogger(__name__)

### Unique RUN_ID for this execution
RUN_ID = str(time.time())

timings = {}
timings["train_cost"] = []
timings["valid_cost"] = []
###

def save(model):
    print "Saving the model..."

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    model.save(model.state['save_dir'] + '/' + RUN_ID + "_" + model.state['prefix'] + 'model.npz')
    cPickle.dump(model.state, open(model.state['save_dir'] + '/' +  RUN_ID + "_" + model.state['prefix'] + 'state.pkl', 'w'))
    numpy.savez(model.state['save_dir'] + '/' + RUN_ID + "_" + model.state['prefix'] + 'timing.npz', **timings)
    signal.signal(signal.SIGINT, s)
    
    print "Model saved, took {}".format(time.time() - start)

def load(model, filename):
    print "Loading the model..."

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    model.load(filename)
    signal.signal(signal.SIGINT, s)

    print "Model loaded, took {}".format(time.time() - start)

def main(args):     
    global timings
    state = eval(args.prototype)() 
    model_type = args.model

    logging.basicConfig(level=getattr(logging, state['level']), \
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    state['model_id'] = RUN_ID
    state['prefix'] = model_type + "_" + state['prefix']
    
    if args.resume != "":
        logger.debug("Resuming %s" % args.resume)
        
        state_file = args.resume + '_state.pkl'
        timings_file = args.resume + '_timing.npz'
        
        if os.path.isfile(state_file):
            logger.debug("Loading previous state")
            
            state = cPickle.load(open(state_file, 'r'))
            timings = dict(numpy.load(open(timings_file, 'r')))
            timings['train_cost'] = list(timings['train_cost'])
            timings['valid_cost'] = list(timings['valid_cost'])
        else:
            raise Exception("Cannot resume, cannot find files!")
    
    logger.debug("State:\n{}".format(pprint.pformat(state)))
    logger.debug("Timings:\n{}".format(pprint.pformat(timings)))
    logger.debug("Compile trainer")
    logger.debug(str(timings)) 
    model = SessionEncoderDecoder(state)
    
    train_batch = model.build_train_function()
    eval_batch = model.build_eval_function()
    sample = model.build_sampling_function()

    if args.resume != "":
        filename = args.resume + '_model.npz'
        if os.path.isfile(filename):
            logger.debug("Loading previous model")
            load(model, filename)

    logger.debug("Load data")
    train_data, \
    valid_data = get_batch_iterator(state)
    
    train_data.start()

    # Start looping through the dataset
    step = 0
    patience = state['patience'] 
    start_time = time.time()
     
    old_valid_cost = 1e21
    train_cost = 0
    train_done = 0
    ex_done = 0
     
    while (step < state['loopIters'] and
            (time.time() - start_time)/60. < state['timeStop'] and
            patience >= 0):
        
        # Sample stuff
        if step % 200 == 0:
            for param in model.params:
                print "%s = %.4f" % (param.name, numpy.sum(param.get_value() ** 2) ** 0.5)
             
            samples, log_probs = sample(1, 40)
            print "Sampled : {}".format(" | ".join(model.indices_to_words(numpy.ravel(samples))))
         
        # Training phase
        batch = train_data.next() 
        # Train finished
        if not batch:
            # Restart training
            logger.debug("Got None...")
            break
        
        logger.debug("[TRAIN] - Got batch %d,%d" % (batch['x'].shape[1], batch['max_length']))
        
        x_data = batch['x']
        x_ranks = batch['y']
        max_length = batch['max_length']
        x_cost_mask = batch['x_mask']

        c = train_batch(x_data, x_ranks, max_length, x_cost_mask)
        if numpy.isinf(c) or numpy.isnan(c):
            logger.warn("Got NaN cost .. skipping")
            continue

        train_cost += c
        train_done += batch['num_preds']

        this_time = time.time()
        if step % state['trainFreq'] == 0:
            elapsed = this_time - start_time
            h, m, s = ConvertTimedelta(this_time - start_time)
            print ".. %.2d:%.2d:%.2d %4d mb # %d bs %d maxl %d acc_cost = %.4f" % (h, m, s,\
                                                                             state['timeStop'] - (time.time() - start_time)/60.,\
                                                                             step, \
                                                                             batch['x'].shape[1], \
                                                                             batch['max_length'], \
                                                                             float(train_cost/train_done))        
        if valid_data is not None and\
            step % state['validFreq'] == 0 and\
                step > 1:
                
                valid_data.start()
                valid_cost = 0
                valid_done = 0
                rank_cost = 0

                logger.debug("[VALIDATION START]") 
                while True:
                    batch = valid_data.next()
                    # Train finished
                    if not batch:
                        break
                     
                    logger.debug("[VALID] - Got batch %d,%d" % (batch['x'].shape[1], batch['max_length']))
        
                    x_data = batch['x']
                    x_ranks = batch['y']
                    max_length = batch['max_length']
                    x_cost_mask = batch['x_mask']

                    pc, rc, mc = eval_batch(x_data, x_ranks, max_length, x_cost_mask)
                    if numpy.isinf(pc) or numpy.isnan(pc):
                        continue
                    
                    valid_cost += pc
                    rank_cost += rc
                    valid_done += batch['num_preds']
                 
                logger.debug("[VALIDATION END]") 
                 
                valid_cost /= valid_done
                rank_cost /= valid_done

                if valid_cost >= old_valid_cost * state['cost_threshold']:
                    patience -= 1
                elif valid_cost < old_valid_cost:
                    patience = state['patience']
                    old_valid_cost = valid_cost
                    # Saving model if decrease in validation cost
                    save(model)
                
                print "** validation error = %.4f, rank error = %.4f, patience = %d" % \
                    (float(valid_cost), float(rank_cost), patience)
                
                timings["train_cost"].append(train_cost/train_done)
                timings["valid_cost"].append(valid_cost)
                
                # Reset train cost and train done
                train_cost = 0
                train_done = 0
        
        step += 1

    logger.debug("All done, exiting...")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="", help="Resume training from that state")
    parser.add_argument("--model", type=str, default='ae', help="Train a model anew of this type (LM or AE)")
    parser.add_argument("--prototype", type=str, help="Use the prototype", default='prototype_rfp')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
