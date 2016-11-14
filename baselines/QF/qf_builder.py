"""
__author__ Alessandro Sordoni
"""

import logging
import cPickle
import os
import sys
import argparse
import numpy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def build(session_file):
    f = open(session_file, 'r')
    
    query_freq = {}
    total_freq = 0
    for num, session in enumerate(f):
        session = session.strip().split('\t')
        for query in session:
            query_freq[query] = query_freq.get(query, 0.) + 1.
            total_freq += 1
         
        if num % 100000 == 0:
            logger.info('{} sessions / {} nodes in the PST'.format(num, len(query_freq)))
    f.close() 
    
    logger.info('-- Closing')
     
    cPickle.dump(query_freq, open(session_file + '_QF.mdl', 'w')) 
