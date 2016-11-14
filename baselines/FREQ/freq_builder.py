"""
__author__ Alessandro Sordoni
"""

import logging
import cPickle
import os
import sys
import argparse
import collections
import operator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def build(session_file):
    # Count absolute query frequency in the training data
    logger.info('Gathering frequency statistics ...')
    
    freq_dict = collections.Counter()
    train_file = open(session_file, 'r')
    for num, line in enumerate(train_file):
        if num % 1000 == 0:
            logger.info('{} sessions / {} queries'.format(num, len(freq_dict)))
     
        queries = line.strip().split('\t')
        for query in queries:
            freq_dict[query] = freq_dict.get(query, 0) + 1
    train_file.close()
    
    logger.info('{} sessions / {} queries'.format(num + 1, len(freq_dict)))
    cPickle.dump(freq_dict, open(session_file + '_FREQ.mdl', 'w'))
