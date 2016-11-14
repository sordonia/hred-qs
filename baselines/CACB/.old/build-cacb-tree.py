"""
__author__ Alessandro Sordoni
"""

import logging
import cPickle
import os
import sys
import argparse
import cacb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('cluster_file', type=str, help='Pickled cluster file')
    parser.add_argument('sessions', type=str, help='Session file')
    args = parser.parse_args()

    assert os.path.isfile(args.sessions) \
            and os.path.isfile(args.cluster_file)
    
    cacbt = cacb.CACB(4)
    cacbt.with_cluster(args.cluster_file)

    cluster_coverage = 0
    f = open(args.sessions, 'r')
    for num, session in enumerate(f):
        cluster_coverage += cacbt.add_session(session)
        
        if num % 1000 == 0:
            print '{} sessions / {} cc / {} nodes in the PST'.format(num, cluster_coverage, cacbt.num_nodes)
    f.close()
    
    # print '{} nodes final'.format(cacbt.num_nodes)
    cacbt.prune()
    # print '{} nodes after pruning'.format(cacbt.num_nodes)
    cacbt.save(args.sessions + "_CACB.pkl")
