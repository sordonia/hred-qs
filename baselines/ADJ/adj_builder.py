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
logger.setLevel(logging.INFO)

def build(session_file):
    # Adjacency, D = 1
    pstree = PST(D=1)
    f = open(session_file, 'r')

    for num, session in enumerate(f):
        pstree.add_session(session.strip().split("\t"))
        if num % 10000 == 0:
            logger.info('{} sessions / {} nodes in the PST'.format(num, pstree.num_nodes))
    f.close()
    logger.info('-- Closing')
    logger.info('{} sessions / {} nodes in the PST'.format(num, pstree.num_nodes))
    pstree.save(session_file + '_ADJ.mdl')
