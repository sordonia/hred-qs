#!/usr/bin/env python
"""
Sampling (BEAM SEARCH) code.
The code is inspired from nmt code implemented using Groundhog
library available at => https://github.com/lisa-groundhog/GroundHog/
However, this code does not necessitate the Groundhog library to run.
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Alessandro Sordoni")
__contact__ = "Alessandro Sordoni <sordonia@iro.umontreal>"

import argparse
import cPickle
import traceback
import logging
import time
import sys

import os
import numpy
import codecs
import itertools
import operator

from search import BeamSampler
from session_encdec import SessionEncoderDecoder
from numpy_compat import argpartition
from state import prototype_state

logger = logging.getLogger(__name__)

class Timer(object):
    def __init__(self):
        self.total = 0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total += time.time() - self.start_time

def parse_args():
    parser = argparse.ArgumentParser("Sample (with beam-search) from the session model")
    parser.add_argument(
        "--n-samples", default="1", type=int,
        help="If used with --beam-search, the size of the beam")
    parser.add_argument("--session", action="store_true", default=False)
    parser.add_argument(
        "--ignore-unk", default=True, action="store_true",
        help="Ignore unknown words")
    parser.add_argument(
        "model_prefix",
        help="Path to the model prefix (without _model.npz or _state.pkl)")
    parser.add_argument("--changes", type=str)
    parser.add_argument(
        "--normalize", action="store_true", default=False,
        help="Normalize log-prob with the word count")
    parser.add_argument("ext_file", nargs="*", default="", help="Changes to state")
    return parser.parse_args()

def print_output_suggestions(output_path, seq, sugg_text, sugg_ranks, sugg_costs):
    assert len(sugg_ranks) == len(sugg_costs)
    assert len(sugg_text) == len(sugg_ranks)
    lambdas = [0., 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    for lambda_param in lambdas:
        output_text = open(output_path + "{}_HAE.gen".format(lambda_param), "w")
        for i, sugg_i in enumerate(sugg_text):
            cost_i = sugg_costs[i][:10]
            rank_i = sugg_ranks[i][:10]
            cost = []
            for c, r in zip(cost_i, rank_i):
                cost.append(c + lambda_param * (r**2))
            best_sugg = map(lambda x : x[1], sorted(zip(cost, sugg_i), key=operator.itemgetter(0)))
            # Make sure that we do not generate the same
            # query, it could happen
            last_ctx = seq[i].strip().split('\t')[-1]
            best_sugg = [x for x in best_sugg if len(x) > 0 if x != last_ctx]

            print >> output_text, '\t'.join(best_sugg)
        output_text.close()

def main():
    args = parse_args()
    state = prototype_state()
    seqs = [[]]
    state_path = args.model_prefix + "_state.pkl"
    model_path = args.model_prefix + "_model.npz"

    with open(state_path) as src:
        state.update(cPickle.load(src))
    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    model = SessionEncoderDecoder(state)
    if os.path.isfile(model_path):
        logger.debug("Loading previous model")
        model.load(model_path)
    else:
        raise Exception("Must specify a valid model path")
    beam_search = BeamSampler(model)
    beam_search.compile()
    for ctx_file in args.ext_file:
        lines = open(ctx_file, "r").readlines()
        seqs = context_to_indices(lines, model)
        sugg_text, sugg_ranks, sugg_costs = \
            sample(model, seqs=seqs, ignore_unk=args.ignore_unk,
                   beam_search=beam_search, n_samples=args.n_samples, session=args.session)
        output_path = ctx_file + "_" + model.state['model_id']
        print_output_suggestions(output_path, lines, sugg_text, sugg_ranks, sugg_costs)

if __name__ == "__main__":
    main()
