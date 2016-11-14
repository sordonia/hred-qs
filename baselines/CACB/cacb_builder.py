"""
__author__ Alessandro Sordoni
"""

import logging
import cPickle
import os
import sys
import argparse
import operator
import cacb
import cluster

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def build_cluster(log_file, cluster_file):
    graph, query_dict, url_dict, click_dict = cluster.build_graph(log_file)
    cluster_ids = cluster.build_cluster(graph, len(url_dict))
    inv_query_dict = sorted(query_dict.items(), key=operator.itemgetter(1))
    
    # Save clusters, one line per cluster 
    cluster_dump = {}
    for cluster_num, qid_list in enumerate(cluster_ids):
        # Put into order of most clicked -> least clicked
        q_r = [(inv_query_dict[qid][0], click_dict[qid]) for qid in qid_list]
        q_r = sorted(q_r, key=operator.itemgetter(1), reverse=True)
        queries, ranks = map(list, zip(*q_r))
        cluster_dump['c_{}'.format(cluster_num)] = {'queries' : queries, 'ranks' : ranks}
     
    cPickle.dump(cluster_dump, open(cluster_file, 'w'))

def build_cacb(cluster_file, session_file):
    cacbt = cacb.CACB()
    cacbt.with_cluster(cluster_file)

    cluster_coverage = 0
    f = open(session_file, 'r')
    for num, session in enumerate(f):
        cluster_coverage += cacbt.add_session(session.strip().split("\t"))
        
        if num % 100000 == 0:
            logger.info(' - {} sessions / {} cc / {} nodes in the PST'.format(num, cluster_coverage, cacbt.num_nodes))
    f.close()
     
    logger.info(' - {} nodes final'.format(cacbt.num_nodes))
    # cacbt.prune()
    # logger.info('{} nodes after pruning'.format(cacbt.num_nodes))
    cacbt.save(session_file + "_CACB.mdl")

def build(log_file, session_file):
    cluster_file = log_file + '.cacb-clst.pkl'
    if os.path.isfile(cluster_file):
        logger.info('Cluster file already detected skipping')
    else:
        build_cluster(log_file, cluster_file)
    build_cacb(cluster_file, session_file)
