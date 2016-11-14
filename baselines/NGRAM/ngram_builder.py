"""
__author__ Alessandro Sordoni
"""

import logging
import cPickle
import os
import sys
import argparse
from Common.psteff import *

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def build(session_file):
    pstree = PST(D=10) 
    f = open(session_file, 'r')

    for num, session in enumerate(f):
        pstree.add_session(session.strip().split("\t"))
        
        if num % 100000 == 0:
            logger.info('{} sessions / {} nodes in the PST'.format(num, pstree.num_nodes))
    
    f.close() 
    pstree.save(session_file + '_NGRAM.mdl')
