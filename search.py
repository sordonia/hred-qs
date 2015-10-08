#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys

import os
import numpy
import codecs

from numpy_compat import argpartition
from state import prototype_state
logger = logging.getLogger(__name__)

def sample_wrapper(sample_logic):
    def sample_apply(*args, **kwargs):
        sampler = args[0]
        contexts = args[1]

        verbose = kwargs.get('verbose', False)

        if verbose:
            logger.info("Starting {} : {} start sequences in total".format(sampler.name, len(contexts)))
        context_samples = []
        context_costs = []

        # Start loop for each sentence
        for context_id, context_sentences in enumerate(contexts):
            # Convert contextes into list of ids
            joined_context = [sampler.model.eoq_sym]
            for num, sentence in enumerate(context_sentences):
                sentence_ids = sampler.model.words_to_indices(sentence.split())
                joined_context += sentence_ids
                if num != len(context_sentences) - 1 or not kwargs.get('complete', False):
                    joined_context += [sampler.model.eoq_sym]

            if verbose:
                logger.info("Searching for {}".format(context_sentences))
                logger.info("Converted into {}".format(joined_context))
            samples, costs = sample_logic(sampler, joined_context, **kwargs)
            # Convert back indices to list of words
            converted_samples = map(lambda sample : sampler.model.indices_to_words(sample), samples)
            # Join the list of words
            converted_samples = map(' '.join, converted_samples)

            if verbose:
                for i in range(len(converted_samples)):
                    print "{}: {}".format(costs[i], converted_samples[i].encode('utf-8'))

            context_samples.append(converted_samples)
            context_costs.append(costs)

        return context_samples, context_costs
    return sample_apply

class Sampler(object):
    def __init__(self, model):
        self.name = 'Sampler'
        self.model = model
        self.compiled = False

    def select_next_words(self, next_probs, step_num, how_many):
        pass

    def compile(self):
        self.next_probs_predictor = self.model.build_next_probs_function()
        self.compute_encoding = self.model.build_encoder_function()
        compiled = True

    def count_n_turns(self, sentence):
        return len([w for w in sentence \
                    if w == self.model.eoq_sym])

    @sample_wrapper
    def sample(self, *args, **kwargs):
        if not self.compiled:
            self.compile()

        context = args[0]

        n_samples = kwargs.get('n_samples', 1)
        ignore_unk = kwargs.get('ignore_unk', True)
        min_length = kwargs.get('min_length', 1)
        max_length = kwargs.get('max_length', 100)
        beam_diversity = kwargs.get('beam_diversity', 1)
        normalize_by_length = kwargs.get('normalize_by_length', True)
        verbose = kwargs.get('verbose', False)
        n_turns = kwargs.get('n_turns', 1)
        complete = kwargs.get('complete', False)

        # Convert to matrix, each column is a context
        # [[1,1,1], [4,4,4], [2,2,2]]
        context = numpy.repeat(
            numpy.array(context, dtype='int32')[:,None], n_samples, axis=1)
        if context[-1, 0] != self.model.eoq_sym and not complete:
            raise Exception('Last token of context, when present,'
                            'should be the end of sentence: %d' % self.model.eoq_sym)

        prev_hs = None
        prev_hd = numpy.zeros((n_samples, self.model.qdim), dtype="float32")
        fin_gen = []
        fin_costs = []
        gen = [[] for i in range(n_samples)]
        costs = [0. for i in range(n_samples)]
        beam_empty = False

        for k in range(max_length):
            if len(fin_gen) >= n_samples or beam_empty:
                break
            if verbose:
                logger.info("{} : sampling step {}, beams alive {}".format(self.name, k, len(gen)))
            # Here we aggregate the context and recompute the hidden state
            # at both session level and query level.
            # Stack only when we sampled something
            if k > 0:
                context = numpy.vstack([context, \
                                        numpy.array(map(lambda g: g[-1], gen))]).astype('int32')
            prev_words = context[-1, :]
            # Recompute hs only for those particular sentences
            # that met the end-of-sentence token
            indx_update_hs = [num for num, prev_word in enumerate(prev_words)
                              if prev_word == self.model.eoq_sym or k == 0]
            if len(indx_update_hs):
                encoder_states = self.compute_encoding(
                    context[:, indx_update_hs])
                if prev_hs is None:
                    prev_hs = encoder_states[2][-1]
                else:
                    prev_hs[indx_update_hs] = encoder_states[2][-1]

            assert prev_hs.ndim == 2
            assert prev_hd.ndim == 2
            assert prev_words.ndim == 1
            next_probs, new_hd = self.next_probs_predictor(
                prev_hs, prev_words, prev_hd)

            assert next_probs.shape[1] == self.model.idim
            # Adjust log probs according to search restrictions
            if ignore_unk:
                next_probs[:, self.model.unk_sym] = 0
            if k <= min_length:
                next_probs[:, self.model.eos_sym] = 0
                next_probs[:, self.model.eoq_sym] = 0

            # Update costs
            next_costs = numpy.array(costs)[:, None] - numpy.log(next_probs)
            # Select next words here
            (beam_indx, word_indx), costs = self.select_next_words(next_costs, next_probs, k, n_samples)
            # Update the stacks
            new_gen = []
            new_costs = []
            new_sources = []

            for num, (beam_ind, word_ind, cost) in enumerate(zip(beam_indx, word_indx, costs)):
                if len(new_gen) > n_samples:
                    break

                hypothesis = gen[beam_ind] + [word_ind]
                # End of query has been detected
                n_turns_hypothesis = self.count_n_turns(hypothesis)
                if n_turns_hypothesis == n_turns:
                    if verbose:
                        logger.debug("adding sentence {} from beam {}".format(hypothesis, beam_ind))

                    # We finished sampling
                    fin_gen.append(hypothesis)
                    fin_costs.append(cost)
                else:
                    # Hypothesis recombination
                    # TODO: pick the one with lowest cost
                    has_similar = False
                    if self.hyp_rec > 0:
                        has_similar = len([g for g in new_gen if \
                            g[-self.hyp_rec:] == hypothesis[-self.hyp_rec:]]) != 0
                    if not has_similar:
                        new_sources.append(beam_ind)
                        new_gen.append(hypothesis)
                        new_costs.append(cost)
            if verbose:
                for gen in new_gen:
                    logger.debug("partial -> {}".format(' '.join(self.model.indices_to_words(gen))))

            prev_hs = prev_hs[new_sources]
            prev_hd = new_hd[new_sources]
            context = context[:, new_sources]
            gen = new_gen
            costs = new_costs
            beam_empty = len(gen) == 0

        # If we have not sampled anything
        # then force include stuff
        if len(fin_gen) == 0:
            fin_gen = gen
            fin_costs = costs
        # Normalize costs
        if normalize_by_length:
            fin_costs = [(fin_costs[num]/len(fin_gen[num])) \
                         for num in range(len(fin_gen))]

        fin_gen = numpy.array(fin_gen)[numpy.argsort(fin_costs)]
        fin_costs = numpy.array(sorted(fin_costs))
        return fin_gen[:n_samples], fin_costs[:n_samples]

class RandomSampler(Sampler):
    def __init__(self, model):
        Sampler.__init__(self, model)
        self.name = 'RandomSampler'
        self.hyp_rec = 0

    def select_next_words(self, next_costs, next_probs, step_num, how_many):
        """ Random sampler just sample a word from the next_probs
            distribution without caring for next costs.
        """
        next_probs = next_probs.astype("float64")
        word_indx = numpy.array([self.model.rng.choice(self.model.idim, p = x/numpy.sum(x))
                                    for x in next_probs], dtype='int32')
        beam_indx = range(next_probs.shape[0])

        args = numpy.ravel_multi_index(numpy.array([beam_indx, word_indx]), next_costs.shape)
        return (beam_indx, word_indx), next_costs.flatten()[args]

class BeamSampler(Sampler):
    def __init__(self, model):
        Sampler.__init__(self, model)
        self.name = 'BeamSampler'
        self.hyp_rec = 3

    def select_next_words(self, next_costs, next_probs, step_num, how_many):
        """ In BeamSampler we pick how_many words for which
            next cost is minimum. Each row of next_costs is a different
            beam. We pick how_many from each beam (how_many * beam_size)
            and then only keep how_many of them.
        """
        flat_next_costs = next_costs.flatten()
        voc_size = next_costs.shape[1]
        args = numpy.argpartition(flat_next_costs, how_many)[:how_many]
        args = args[numpy.argsort(flat_next_costs[args])]
        return numpy.unravel_index(args, next_costs.shape), \
            flat_next_costs[args]
