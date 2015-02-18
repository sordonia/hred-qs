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

class BeamSearch(object):
    def __init__(self, model):
        self.model = model 
        state = self.model.state 
        self.unk_sym = self.model.unk_sym
        self.eos_sym = self.model.eos_sym
        self.eoq_sym = self.model.eoq_sym
        self.qdim = self.model.qdim
        self.sdim = self.model.sdim

    def compile(self):
        logger.debug("Compiling beam search functions")
        self.next_probs_predictor = self.model.build_next_probs_function()
        self.compute_encoding = self.model.build_encoder_function()
        self.rank_prediction = self.model.build_rank_prediction_function()

    def search(self, seq, n_samples=1, ignore_unk=False, minlen=1, normalize_by_length=True, session=False):
        # Make seq a column vector
        def _is_finished(beam_gen):
            if session and beam_gen[-1] == self.eos_sym:
                return True
            if not session and beam_gen[-1] == self.eoq_sym:
                return True
            return False

        seq = numpy.array(seq)
        
        if seq.ndim == 1:
            seq = numpy.array([seq], dtype='int32').T
        else:
            seq = seq.T

        assert seq.ndim == 2
        h, hr, hs = self.compute_encoding(seq)
         
        # Initializing starting points with the last encoding of the sequence 
        prev_words = numpy.zeros((seq.shape[1],), dtype='int32') + self.eoq_sym
        prev_hd = numpy.zeros((seq.shape[1], self.qdim), dtype='float32')
        prev_hs = numpy.zeros((seq.shape[1], self.sdim), dtype='float32')
         
        prev_hs[:] = hs[-1]
         
        fin_beam_gen = []
        fin_beam_costs = []
        fin_beam_ranks = []

        beam_gen = [[]] 
        costs = [0.0]

        max_step = 30
        for k in range(max_step):
            logger.info("Beam search at step %d" % k)
            if n_samples == 0:
                break

            # prev_hd = prev_hd[:beam_size]
            # prev_hs = prev_hs[:beam_size]
            beam_size = len(beam_gen)
            prev_words = (numpy.array(map(lambda bg : bg[-1], beam_gen))
                    if k > 0
                    else numpy.zeros(1, dtype="int32") + self.eoq_sym)
            
            assert prev_hs.shape[0] == prev_hd.shape[0]
            assert prev_words.shape[0] == prev_hs.shape[0]
            
            repeat = numpy.repeat(seq, beam_size, axis=1)
            whole_context = numpy.vstack([repeat, numpy.array(beam_gen,dtype='int32').T])
            h, hr, hs = self.compute_encoding(whole_context)

            outputs, hd = self.next_probs_predictor(hs[-1], prev_words, prev_hd)
            log_probs = numpy.log(outputs)

            # Adjust log probs according to search restrictions
            if ignore_unk:
                log_probs[:, self.unk_sym] = -numpy.inf

            if k <= minlen:
                log_probs[:, self.eos_sym] = -numpy.inf
                log_probs[:, self.eoq_sym] = -numpy.inf 
            
            # Artificially not reproduce same words
            for i in range(n_samples):
                if k > 0:
                    log_probs[i, beam_gen[i][1:]] = -numpy.inf

            # Find the best options by calling argpartition of flatten array
            next_costs = numpy.array(costs)[:, None] - log_probs
            flat_next_costs = next_costs.flatten()
            best_costs_indices = argpartition(
                    flat_next_costs.flatten(),
                    n_samples)[:n_samples]

            # Decypher flatten indices
            voc_size = log_probs.shape[1]
            trans_indices = best_costs_indices / voc_size
            word_indices = best_costs_indices % voc_size
            costs = flat_next_costs[best_costs_indices]

            # Form a beam for the next iteration
            new_beam_gen = [[]] * n_samples 
            new_costs = numpy.zeros(n_samples)
            
            new_prev_hs = numpy.zeros((n_samples, self.sdim), dtype="float32")
            new_prev_hs[:] = hs[-1]
            new_prev_hd = numpy.zeros((n_samples, self.qdim), dtype="float32")
            
            for i, (orig_idx, next_word, next_cost) in enumerate(
                    zip(trans_indices, word_indices, costs)):
                
                new_beam_gen[i] = beam_gen[orig_idx] + [next_word]
                new_costs[i] = next_cost
                new_prev_hd[i] = hd[orig_idx]
             
            beam_gen = []
            costs = []
            indices = []

            for i in range(n_samples):
                # We finished sampling?
                if not _is_finished(new_beam_gen[i]): 
                    beam_gen.append(new_beam_gen[i])
                    costs.append(new_costs[i])
                    indices.append(i)
                else:
                    n_samples -= 1
                    
                    # Concatenate sequence and predict rank 
                    concat_seq = numpy.vstack([seq, numpy.array([new_beam_gen[i]], dtype='int32').T])
                    ranks = self.rank_prediction(concat_seq)
                    
                    fin_beam_ranks.append(numpy.ravel(ranks)[-1])
                    fin_beam_gen.append(new_beam_gen[i])
                    if normalize_by_length:
                        fin_beam_costs.append(new_costs[i]/len(new_beam_gen[i]))

            # Filter out the finished states 
            prev_hd = new_prev_hd[indices]
            prev_hs = new_prev_hs[indices]
         
        fin_beam_gen = numpy.array(fin_beam_gen)[numpy.argsort(fin_beam_costs)]
        fin_beam_ranks = numpy.array(fin_beam_ranks)[numpy.argsort(fin_beam_costs)]
        fin_beam_costs = numpy.array(sorted(fin_beam_costs))
         
        return fin_beam_gen, fin_beam_costs, fin_beam_ranks

def sample(model, seqs=[[]], n_samples=1, beam_search=None, ignore_unk=False, normalize=False, session=False): 
    if beam_search:
        logger.info("Starting beam search : {} start sequences in total".format(len(seqs)))
        
        seqs_gen = []
        seqs_ranks = []
        seqs_costs = []

        for idx, seq in enumerate(seqs):
            sentences = []

            logger.info("Searching for {}/{}".format(idx, seq))
            
            gen_ranks = []
            gen_costs = []
            gen_queries = []
            
            if len(seq):
                gen_queries, gen_costs, gen_ranks = beam_search.search(seq, n_samples, ignore_unk=ignore_unk, session=session) 
                for i in range(len(gen_queries)):
                    query = model.indices_to_words(gen_queries[i])
                    sentences.append(' | '.join(query))
            
            seqs_gen.append(sentences)
            seqs_ranks.append(gen_ranks)
            seqs_costs.append(gen_costs)

            for i in range(len(gen_costs)):   
                logger.info("{} - {}: {}".format(gen_ranks[i], gen_costs[i], sentences[i].encode('utf-8')))
             
        return seqs_gen, seqs_ranks, seqs_costs 
    else:
        raise Exception("Only beam-search is supported")

def context_to_indices(contexts, model):
    ''' Convert a sequence of sequences to indices '''
    seqs = [[]] * len(contexts)
    for ctx_indx, ctx_text in enumerate(contexts):
        ctx_queries = ctx_text.strip().split('\t')
        # Test if last one contains an unknown word
        # otherwise do not recommend
        indx = model.words_to_indices(ctx_queries[-1].split(), add_se=True)
        if 0 in indx:
            seqs[ctx_indx] = []
        else:
            seqs[ctx_indx] = list(itertools.chain( \
                *[model.words_to_indices(c.split(), add_se=True) for c in ctx_queries]))
    return seqs

def parse_args():
    parser = argparse.ArgumentParser("Sample (with beam-search) from the session model")
    
    parser.add_argument("--n-samples", 
            default="1", type=int, 
            help="Number of samples, if used with --beam-search, the size of the beam")
    
    parser.add_argument("--session",
                        action="store_true",
                        default=False)

    parser.add_argument("--ignore-unk",
            default=True, action="store_true",
            help="Ignore unknown words")
    
    parser.add_argument("model_prefix",
            help="Path to the model prefix (without _model.npz or _state.pkl)")
    
    parser.add_argument("--changes", type=str)

    parser.add_argument("--normalize",
            action="store_true", default=False,
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
     
    beam_search = BeamSearch(model)
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
