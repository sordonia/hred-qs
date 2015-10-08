#!/usr/bin/env python

from data_iterator import *
from state import *
from session_encdec import *
from utils import *
from evaluation import *

import time
import traceback
import os.path
import sys
import argparse
import cPickle
import logging
import search
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
### Additional measures can be set here
measures = ["train", "valid"]

def init_timings():
    timings = {}
    for m in measures:
        timings[m] = []
    return timings

def save(model, timings):
    print "Saving the model..."

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    model.save(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + '_model.npz')
    cPickle.dump(model.state, open(model.state['save_dir'] + '/' +  model.state['run_id'] + "_" + model.state['prefix'] + '_state.pkl', 'w'))
    numpy.savez(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + '_timing.npz', **timings)
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
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    state = eval(args.prototype)()
    timings = init_timings()

    if args.resume != "":
        logger.debug("Resuming %s" % args.resume)
        state_file = args.resume + '_state.pkl'
        timings_file = args.resume + '_timing.npz'
        if os.path.isfile(state_file) and os.path.isfile(timings_file):
            logger.debug("Loading previous state")
            state = cPickle.load(open(state_file, 'r'))
            timings = dict(numpy.load(open(timings_file, 'r')))
            for x, y in timings.items():
                timings[x] = list(y)
        else:
            raise Exception("Cannot resume, cannot find files!")

    logger.info("State:\n{}".format(pprint.pformat(state)))
    logger.info("Timings:\n{}".format(pprint.pformat(timings)))

    model = SessionEncoderDecoder(state)
    rng = model.rng

    if args.resume != "":
        filename = args.resume + '_model.npz'
        if os.path.isfile(filename):
            logger.info("Loading previous model")
            load(model, filename)
        else:
            raise Exception("Cannot resume, cannot find model file!")
    else:
        # assign new run_id key
        model.state['run_id'] = RUN_ID

    logger.info("Compile trainer")
    train_batch = model.build_train_function()
    eval_batch = model.build_eval_function()
    random_sampler = search.RandomSampler(model)

    logger.info("Load data")
    train_data, valid_data = get_batch_iterator(rng, state)
    train_data.start()

    # Start looping through the dataset
    step = 0
    patience = state['patience']
    start_time = time.time()
    train_cost = 0
    train_done = 0
    ex_done = 0
    while step < state['loop_iters'] and patience >= 0:
        # Sample stuff
        if step % 200 == 0:
            for param in model.params:
                print "%s = %.4f" % (param.name,
                                     numpy.sum(param.get_value() ** 2) ** 0.5)
            samples, costs = random_sampler.sample([[]], n_samples=1, n_turns=3)
            print "Sampled : {}".format(samples[0])

        # Training phase
        batch = train_data.next()
        # Train finished
        if not batch:
            # Restart training
            logger.debug("Got None...")
            break
        c = train_batch(
            batch['x'], batch['y'], batch['max_length'], batch['x_mask'])
        if numpy.isinf(c) or numpy.isnan(c):
            logger.warn("Got NaN cost .. skipping")
            continue

        train_cost += c
        train_done += batch['num_preds']

        this_time = time.time()
        if step % state['train_freq'] == 0:
            elapsed = this_time - start_time
            h, m, s = ConvertTimedelta(this_time - start_time)
            print ".. %.2d:%.2d:%.2d %4d mb # %d bs %d maxl %d acc_cost = %.4f" % (h, m, s,\
                                                                             state['time_stop'] - (time.time() - start_time)/60.,\
                                                                             step, \
                                                                             batch['x'].shape[1], \
                                                                             batch['max_length'], \
                                                                             float(train_cost/train_done))
        if valid_data is not None and\
            step % state['valid_freq'] == 0 and step > 1:
                valid_data.start()
                valid_cost = 0
                valid_done = 0
                logger.debug("[VALIDATION START]")
                while True:
                    batch = valid_data.next()
                    # Train finished
                    if not batch:
                        break
                    if numpy.isinf(c) or numpy.isnan(c):
                        continue
                    c = eval_batch(
                        batch['x'], batch['y'], batch['max_length'], batch['x_mask'])
                    valid_cost += c
                    valid_done += batch['num_preds']
                logger.debug("[VALIDATION END]")
                valid_cost /= valid_done
                if len(timings["valid"]) == 0 or valid_cost < numpy.min(numpy.array(timings["valid"])):
                    patience = state['patience']
                    # Saving model if decrease in validation cost
                    save(model, timings)
                elif valid_cost >= timings["valid"][-1] * state['cost_threshold']:
                    patience -= 1

                print "** validation error = %.4f, patience = %d" % (float(valid_cost), patience)
                timings["train"].append(train_cost/train_done)
                timings["valid"].append(valid_cost)

                # Reset train cost and train done
                train_cost = 0
                train_done = 0
        step += 1
    logger.debug("All done, exiting...")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="", help="Resume training from that state")
    parser.add_argument("--prototype", type=str, help="Use the prototype", default='prototype_state')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
