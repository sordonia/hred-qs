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

def build(session_file, epsilon):
    D = 5

    pstree = PST(D) 
    f = open(session_file, 'r')
    
    for num, session in enumerate(f):
        pstree.add_session(session.strip().split("\t"))
        
        if num % 100000 == 0:
            logger.info('{} sessions / {} nodes in the PST'.format(num, pstree.num_nodes))
    f.close()
  
    if epsilon != 0.:
        logger.info("Pruning with epsilon = {}".format(epsilon))
        pstree.prune(epsilon=epsilon)
        logger.info("End Pruning.")
    
    logger.info('-- Closing')
    logger.info('{} sessions / {} nodes in the PST'.format(num, pstree.num_nodes))
    
    pstree.save(session_file + ('_e{}'.format(epsilon) if epsilon != 0 else '') + '_VMM.mdl')

    #print 'Testing loading'
    #pstree.load(args.output_prefix + '_pst_e{}_d{}.pkl'.format(epsilon, D))
    #print '{} nodes'.format(pstree.num_nodes)
