"""
Takes as input a session file and a rank file and creates a processed version of it.
If given an external dictionary, the input files will be converted
using that input dictionary.

@author Alessandro Sordoni
"""

import collections
import numpy
import operator
import os
import sys
import logging
import cPickle
import itertools
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('text2dict')

def safe_pickle(obj, filename):
    if os.path.isfile(filename):
        logger.info("Overwriting %s." % filename)
    else:
        logger.info("Saving to %s." % filename)
    
    with open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Prefix (*.ses, *.rnk) to separated session/rank file")
parser.add_argument("--cutoff", type=int, default=-1, help="Vocabulary cutoff (optional)")
parser.add_argument("--min_freq", type=int, default=1, help="Min frequency cutoff (optional)")
parser.add_argument("--dict", type=str, default="", help="External dictionary (pkl file)")
parser.add_argument("output", type=str, help="Output file") 
args = parser.parse_args()

if not os.path.isfile(args.input + ".ses") or not os.path.isfile(args.input + ".rnk"):
    raise Exception("Input file not found!")

unk = "<unk>"

###############################
# Part I: Create the dictionary
###############################
if args.dict != "":
    # Load external dictionary
    assert os.path.isfile(args.dict)
    vocab = dict([(x[0], x[1]) for x in cPickle.load(open(args.dict, "r"))])
    
    # Check consistency
    assert '<unk>' in vocab
    assert '<q>' in vocab
    assert '</s>' in vocab
    assert '</q>' in vocab
else:
    word_counter = Counter()

    for line in open(args.input + ".ses", 'r'):
        s = [x for x in line.strip().split()]
        word_counter.update(s) 

    total_freq = sum(word_counter.values())
    logger.info("Total word frequency in dictionary %d " % total_freq) 

    if args.cutoff != -1:
        logger.info("Cutoff %d" % args.cutoff)
        vocab_count = word_counter.most_common(args.cutoff)
    else:
        vocab_count = word_counter.most_common()

    # Add special tokens to the vocabulary
    vocab = {'<unk>': 0, '<q>': 1, '</q>': 2, '</s>': 3}
    for (word, count) in vocab_count:
        if count < args.min_freq:
            break
        vocab[word] = len(vocab)

logger.info("Vocab size %d" % len(vocab))

#################################
# Part II: Binarize the triples
#################################

# Everything is loaded into memory for the moment
binarized_corpus = []
binarized_ranks = []

# Some statistics
mean_sl = 0.
unknowns = 0.
num_terms = 0.
freqs = collections.defaultdict(lambda: 1)

for line, (session, rank) in enumerate(itertools.izip(open(args.input + ".ses", 'r'), \
                                                    open(args.input + ".rnk", 'r'))):
    session_lst = []
    rank_lst = []

    queries = session.split('\t')
    ranks = rank.split('\t')

    for i, query in enumerate(queries):
        query_lst = []
        for word in query.strip().split():
            query_lst.append(vocab.get(word, 0))
            unknowns += query_lst[-1] == 0
            freqs[query_lst[-1]] += 1

        num_terms += len(query_lst) 
        
        # Here, we filter out unknown triple text and empty triples
        # i.e. <q> </q> or <q> 0 </q>
        if query_lst != [0] and len(query_lst):
            session_lst.append([1] + query_lst + [2]) 
            freqs[1] += 1
            freqs[2] += 1
            
            rank_lst.append(ranks[i])

    if len(session_lst) > 1:
        rank_lst.append(0)
        session_lst.append([1,3])
        freqs[1] += 1
        freqs[3] += 1

        # Flatten out binarized triple
        # [[a, b, c], [c, d, e]] -> [a, b, c, d, e]
        assert len(rank_lst) == len(session_lst)        
        binarized_ranks.append(rank_lst)
        binarized_session = list(itertools.chain(*session_lst))  
        binarized_corpus.append(binarized_session)

safe_pickle(binarized_corpus, args.output + ".ses.pkl")
safe_pickle(binarized_ranks, args.output + ".rnk.pkl")
if args.dict == "":
     safe_pickle([(word, word_id, freqs[word_id]) for word, word_id in vocab.items()], args.output + ".dict.pkl")

logger.info("Number of unknowns %d" % unknowns)
logger.info("Number of terms %d" % num_terms)
logger.info("Mean session length %f" % float(sum(map(len, binarized_corpus))/len(binarized_corpus)))
logger.info("Writing training %d sessions (%d left out)" % (len(binarized_corpus), line + 1 - len(binarized_corpus)))
