#!/usr/bin/env python

import argparse
import cPickle
import logging
import time
import sys

import itertools
import os
import numpy

from session_encdec import SessionEncoderDecoder
from state import prototype_state

logger = logging.getLogger(__name__)

class IterativeScorer(object):
    def __init__(self, model):
        self.model = model
        state = self.model.state
        self.unk_sym = self.model.unk_sym
        self.eoq_sym = self.model.eoq_sym
        self.eos_sym = self.model.eos_sym
        self.soq_sym = self.model.soq_sym
        self.qdim = self.model.qdim
        self.sdim = self.model.sdim
        self.compiled = False

    def compile(self):
        logger.debug("Compiling scorer functions")
        self.score_fn = self.model.build_score_function()
        self.compiled = True

    def score(self, context, targets, verbose=False, normalize_by_length=False):
        if not self.compiled:
            self.compile()

        # Prepare target matrix
        num_tgt = len(targets)
        ctx_length = len(context)

        max_length = numpy.max(map(lambda x : len(x), targets))
        x_data = numpy.zeros((ctx_length + max_length, num_tgt), dtype='int32')

        for x in range(num_tgt):
           x_data[:(ctx_length + len(targets[x])), x] \
                   = numpy.array(context + targets[x], dtype='int32')

        costs = [0.0 for i in range(num_tgt)]
        log_probs = self.score_fn(x_data, max_length + ctx_length)

        # Cutoff context probs
        log_probs = log_probs[0][ctx_length:,:]
        for x in range(num_tgt):
            len_target = len(targets[x])
            if normalize_by_length:
                costs[x] = numpy.mean(log_probs[:len_target,x])
            else:
                costs[x] = numpy.sum(log_probs[:len_target,x])
        return costs

class Scorer(object):
    """
    A simple scorer class
    """
    def __init__(self, model):
        # Compile beam search
        self.model = model
        self.scorer = IterativeScorer(model)
        self.scorer.compile()

    def score(self, contexts, targets, verbose=False, \
              normalize_by_length=False, N=1):
        if verbose:
            logger.info("Starting scoring: {} start sequences in total".format(len(targets)))

        def _convert_sentence(sentence):
            sentence_ids = self.model.words_to_indices(sentence.split())
            if self.model.soq_sym != -1:
                sentence_ids = [self.model.soq_sym] + sentence_ids
            sentence_ids += [self.model.eoq_sym]
            return sentence_ids

        costs = []
        for num, (context, target) in enumerate(zip(contexts, targets)):
            if num % 100 == 0:
                logger.info("Done {}/{}".format(num, len(contexts)))
            if verbose:
                logger.info("Searching for {}".format(context))

            # Convert contextes into list of ids
            context_sentences_ids = map(_convert_sentence, context)
            joined_contexts = [[] for i in range(N)]
            context_nums = min(len(context_sentences_ids), N)
            for i in range(min(len(context_sentences_ids), N - 1)):
                joined_contexts[i] = list(itertools.chain(*context_sentences_ids[len(context_sentences_ids)-i-1:]))
            if context_nums == N:
                joined_contexts[N-1] = list(itertools.chain(*context_sentences_ids[0:]))
            if verbose:
                logger.info(str(joined_contexts))
            converted_targets = []
            for target_sentence in target:
                sentence_ids = _convert_sentence(target_sentence)
                converted_targets += [sentence_ids + [self.model.eos_sym]]
            joined_costs = []
            for joined_context in joined_contexts:
                if len(joined_context):
                    local_costs = self.scorer.score(
                        joined_context, converted_targets, verbose=verbose,
                        normalize_by_length=normalize_by_length)
                else:
                    local_costs = joined_costs[-1]
                joined_costs.append(local_costs)
            costs.append(joined_costs)
        return costs

def parse_args():
    parser = argparse.ArgumentParser("Score a given ctx file wrt a rnk file")
    parser.add_argument("model_prefix",
            help="Path to the model prefix (without _model.npz or _state.pkl)")
    parser.add_argument("context",
            help="File of input contexts (sessions, tab separated)")
    parser.add_argument("targets",
            help="File of input targets (candidates, tab separated)")
    parser.add_argument("--feature-gen",
            action="store_true", default=False,
            help="Feature generation mode")
    parser.add_argument("--multi-feature", default=1, type=int)
    parser.add_argument("--normalize-by-length",
            action="store_true")
    parser.add_argument("--verbose",
            action="store_true", default=False,
            help="Be verbose")
    parser.add_argument("changes", nargs="?", default="", help="Changes to state")
    return parser.parse_args()

def main():
    args = parse_args()
    state = prototype_state()

    state_path = args.model_prefix + "_state.pkl"
    model_path = args.model_prefix + "_model.npz"

    with open(state_path) as src:
        state.update(cPickle.load(src))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    model = SessionEncoderDecoder(state)
    scorer = Scorer(model)

    if os.path.isfile(model_path):
        logger.debug("Loading previous model")
        model.load(model_path)
    else:
        raise Exception("Must specify a valid model path")

    contexts = [[]]
    lines = open(args.context, "r").readlines()
    contexts = [x.strip().split('\t') for x in lines]

    targets = [[]]
    lines = open(args.targets, "r").readlines()
    targets = [x.strip().split('\t') for x in lines]

    logging.info('Normalizing by length = {}'.format(args.normalize_by_length))
    logging.info('Multi feature = {}'.format(args.multi_feature))

    costs = scorer.score(contexts,
                         targets,
                         verbose=args.verbose,
                         normalize_by_length=args.normalize_by_length,
                         N=args.multi_feature)

    output_handle = open(args.targets + "_HED_" + ("nn_" if not args.normalize_by_length else "") + \
                         model.run_id + (".f" if args.feature_gen else ".gen"), "w")

    if args.feature_gen:
        print >> output_handle, ' '.join(["%d_HED_" % i + model.run_id for i in range(args.multi_feature)])

    for num_target, target in enumerate(targets):
        reranked = numpy.array(target)[numpy.argsort(costs[num_target])]

        if args.feature_gen:
            for cost in numpy.array(costs[num_target]).T:
                print >> output_handle, ' '.join(map(str,cost))
        else:
            print >> output_handle, '\t'.join(reranked)

        output_handle.flush()
    output_handle.close()

if __name__ == "__main__":
    main()
