import argparse
import cPickle
import numpy
import sys
import math
import operator
import itertools
import cluster

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('log_file', type=str, help='the query log')
    args = parser.parse_args()

    graph, query_dict, url_dict, click_dict = cluster.build_graph(args.log_file)
    cluster_ids = cluster.build_cluster(graph, len(url_dict))
    inv_query_dict = sorted(query_dict.items(), key=operator.itemgetter(1))
    
    # Save clusters, one line per cluster 
    cluster_dump = {}
    for cluster_num, qid_list in enumerate(cluster_ids):
        # Put into order of most clicked -> least clicked
        q_r = [(inv_query_dict[qid][0], click_dict[qid]) for qid in qid_list]
        q_r = sorted(q_r, key=operator.itemgetter(1), reverse=True)
        queries, ranks = zip(*q_r)
        cluster_dump['c_{}'.format(cluster_num)] = {'queries' : queries, 'ranks' : ranks}
     
    cPickle.dump(cluster_dump, open(args.log_file + '.cacb-clst.pkl', 'w')) 
