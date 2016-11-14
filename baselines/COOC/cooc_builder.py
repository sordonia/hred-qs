"""
__author__ Alessandro Sordoni
"""

import logging
import cPickle
import os
import sys
import argparse
from Common import pstfast

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def build(session_file):
    # Adjacency, D = 1
    D = 1  
    pstree = pstfast.PST(D) 
    f = open(session_file, 'r')
     
    for num, session in enumerate(f):
        pstree.add_session(session.strip().split("\t"))
        if num % 1000 == 0:
            logger.info('{} sessions / {} nodes in the PST'.format(num, pstree.num_nodes))
     
    f.close()
   
    pstree.save(session_file + '_ADJ.pkl')
