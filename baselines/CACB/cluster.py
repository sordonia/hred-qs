import argparse
import cPickle
import numpy
import sys
import math
import operator
import itertools
import copy
import itertools
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def build_graph(log_file):
    log_in = open(log_file, 'r')
    
    query_to_url = {}
    query_dict = {}
    url_dict = {}
    click_dict = {}
     
    for num, line in enumerate(log_in):
        if num % 1e6 == 0:
            logger.info(' - {} line'.format(num))
        
        fields = line.strip().split('\t')
        if len(fields) < 5:
            continue
         
        query = fields[1]
        url = fields[4]

        if query not in query_dict:
            query_dict[query] = len(query_dict)
        if url not in url_dict:
            url_dict[url] = len(url_dict)

        query = query_dict[query]
        url = url_dict[url]
        click_dict[query] = click_dict.get(query, 0.) + 1. 

        if query not in query_to_url:
            query_to_url[query] = {}
        query_to_url[query][url] = query_to_url[query].get(url, 0.) + 1.
         
    # Prune everything
    logger.info(' - Graph pruning - {} nodes'.format(len(query_to_url)))
    
    for key, url_map in query_to_url.items():
        items = url_map.items()
        allsum = sum(url_map.values())
        
        for url, count in items:
            if count <= 5 or float(count)/allsum <= 0.1:
                del url_map[url]
    
    query_to_url = dict([(key, url_map) for key, url_map in query_to_url.items() if len(url_map) > 0])
    logger.info(' - Pruned - {} nodes'.format(len(query_to_url)))

    for key, url_map in query_to_url.items():
        normalized = normalize_list(url_map.items())
        query_to_url[key] = dict(normalized)
     
    return query_to_url, query_dict, url_dict, click_dict

def normalize_list(list_1):
    norm = math.sqrt(sum([v**2 for x, v in list_1]))
    return [(x, float(v)/float(norm)) for x, v in list_1]

def cluster_center(cluster_size, old_center, new_query):
    keys = set(old_center.keys())
    keys.update(new_query.keys())
    
    new_center = old_center
    for k, v in new_query.items():
        new_center[k] = new_center.get(k, 0) + v
    
    new_center_norm = normalize_list(new_center.items())
    return dict(new_center), dict(new_center_norm) 

def sq_euclid_distance(query_a, query_b):
    keys = set(query_a.keys())
    keys.update(query_b.keys())
    dist = sum([(query_a.get(a, 0) - query_b.get(a, 0)) ** 2 for a in keys])
    return dist

def cluster_radius(old_D, queries_in_cluster):
    C = len(queries_in_cluster)
    
    if C == 1:
        return (0, 0)
    
    D = old_D
    for i in range(len(queries_in_cluster) - 1):
        D += 2 * sq_euclid_distance(queries_in_cluster[i], queries_in_cluster[-1])

    return (D, math.sqrt(D/((C-1)*C)))

def build_cluster(QU, num_url):
    cluster_dim = {}
    cluster_centers = []
    cluster_queries = []
    cluster_ids = []
    cluster_old_diam = []
    cluster_cen_num = []
    cluster_count = 0
    
    start = time.time()
    for num, (key, url_map) in enumerate(QU.items()):
        all_keys = itertools.chain(*[list(cluster_dim[x]) for x in url_map.keys() if x in cluster_dim])
        c_set = set(all_keys)
        
        if len(c_set) > 0:
            min_d = numpy.inf
            min_cid = 0

            for c in list(c_set): 
                dist = math.sqrt(sq_euclid_distance(url_map, cluster_centers[c]))
                if dist < min_d:
                    min_d = dist
                    min_cid = c 
        
            queries_list = [QU[x] for x in cluster_ids[min_cid] + [key]]
            old_D, D = cluster_radius(cluster_old_diam[min_cid], queries_list)
        else:
            D = numpy.inf

        if num % 1000 == 0 and num != 0:
            print "Done query {}/{} - clusters {}".format(num, len(QU), len(cluster_centers))
            end = time.time()
            print "Longest cluster {} - Took {}s".format(max([len(x) for x in cluster_ids]), (end-start))
            print "Candidate set {}".format(len(c_set))
            start = end
            sys.stdout.flush()

        if D <= 1:
            cluster_ids[min_cid].append(key)
            cluster_old_diam[min_cid] = old_D
            cluster_cen_num[min_cid], cluster_centers[min_cid] = cluster_center(len(cluster_ids), cluster_cen_num[min_cid], url_map)
        else:
            cluster_centers.append({})
            cluster_ids.append([])
            cluster_cen_num.append([])
            cluster_old_diam.append(0)

            cluster_ids[cluster_count].append(key)
            cluster_centers[cluster_count] = copy.deepcopy(url_map)
            cluster_cen_num[cluster_count] = copy.deepcopy(url_map)
            min_cid = cluster_count
            cluster_count += 1

        for url_id in url_map.keys():
            if url_id not in cluster_dim:
                cluster_dim[url_id] = set()
            cluster_dim[url_id].add(min_cid)
    
    print 'Finished, {}'.format(len(cluster_centers))
    return cluster_ids
