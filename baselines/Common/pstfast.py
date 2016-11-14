import cPickle
import numpy as np
import os
import sys
import argparse
import collections
import operator
from collections import OrderedDict

import logging
import math

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def kl_divergence(a, b, smoothing):
    norm_a = sum(a.values())
    norm_b = sum(b.values())
     
    # Approximate KL (i.e. smooth while computing)
    kl = 0.
    for x, v in a.items():
        kl += v/norm_a * (np.log(v/norm_a) - np.log(b.get(x, smoothing)/norm_b)) 
    assert kl >= 0
    return kl

class Node(object):
    def __init__(self, query_id=-1):
        # Query id
        self.node_id = query_id
        # Sorted array of children
        self.probs = {}
        self.children = {}

class PSTInfer(object):
    def __init__(self, exact_match=False):
        self.exact_match = exact_match
        self.query_to_id = {}
        self.id_to_query = []    
    
    def _load_pickle(self, input_file):
        f = open(input_file, 'r')     
        self.data = cPickle.load(f)
        self.query_to_id = cPickle.load(f)
        f.close()
 
    def load(self, input_file):
        self._load_pickle(input_file)
        self.id_to_query = [query for query, query_id in \
                            sorted(self.query_to_id.items(), key=operator.itemgetter(1))]
        logger.debug('Loaded inference engine')

    def _find(self, suffix):
        """
        Save the tree in a flattened python format using lists.
        """
        def _get_child(levels, child):
            # levels = [(i, {}), []], [], [], ...
            for level in levels:
                level_root = level[0]
                if level_root[0] == child:
                    return level_root
            return None
       
        suffix = map(lambda x : self.query_to_id.get(x, -1), suffix)
         
        # Initialize node_path with the root
        node_path = [self.data[0]]
        past_node = self.data
         
        # For each query in the suffix, explore the path
        # in the tree
        for query in reversed(suffix):
            if len(past_node) == 0:
                break

            next_node = _get_child(past_node[1:], query)
            if not next_node:
                break
             
            # Append the root of the node in the node path
            node_path.append(next_node)
            past_node = next_node
         
        # If we have found it, then the root
        # correspond to the first element in the suffix
        return node_path, query == suffix[0]
            
    def evaluate(self, suffix, targets):
        node_path, found = _find(suffix)
        targets = set(map(lambda x: self.query_to_id.get(x, -1), targets))

        log_probs = {}
        for node in reversed(node_path):
            probs = dict(node[1])
            for target in targets:
                if target in probs: 
                    log_probs[self.id_to_query[target]] = probs[target]
                    targets.remove(target)

        return log_probs

    def suggest(self, suffix, N=20):
        node_path, is_found = self._find(suffix)
        
        node_id = node_path[-1][0]
        node_scores = node_path[-1][1]

        data = {'last_node_id' : node_id, \
                'last_node_query': self.id_to_query[node_id], \
                'found' : is_found, \
                'suggestions' : [], \
                'scores' : []}
         
        # If it is the root or if we want an exact session 
        # match, i.e. NGRAM models
        if node_id == 0 or (self.exact_match and not found): 
            return data 
        
        # Get top N
        sugg_score = [(self.id_to_query[sugg_id], score) for sugg_id, score in node_scores[:N]]
        sugg, score = map(list, zip(*sugg_score))
         
        data['suggestions'] = sugg
        data['scores'] = score
        return data

class PST(object):
    def __init__(self, D=4):
        self.root = Node(0)
        self.query_dict = {'[ROOT]' : 0}
        self.nodes = [self.root]
        self.D = D
        self.num_nodes = 1

    def __str__(self):
        def _str(node, space=0):
            message = '\t' * space + '@@NODE: %s - @@PROBS: %s' % (node.node_id, self.get_probs(node))
            list_child = node.children.values()

            for child in list_child:
                message += '\n%s' % _str(child, space + 1)
            return message
        return _str(self.root) 
    
    def get_children(self):
        def _get_children(node):
            nodes = [node]
             
            list_child = node.children.values()
            for x in list_child:
                nodes += _get_children(v)
            return nodes
         
        return _get_children(self.root)

    def get_probs(self, node):
        return node.probs.items()
    
    def save(self, output_path):
        """
        Save the tree in a flattened python format using lists.
        [(node_id,{}),[(child_id_1,{q_1 : p_1, q_2 : p_2})],[(child_id_2,{...})]]
        """
        def _flatten(node):
            # Normalize the probabilities when saving the PST
            sum_p = sum(node.probs.values())
            normalized_probs = [(i, float(v)/sum_p) for i, v in node.probs.items()][:20]
             
            sorted_probs = sorted(normalized_probs, key=operator.itemgetter(1), reverse=True)
            
            reprs = [(node.node_id, sorted_probs)]

            list_child = node.children.items()
            for child_id, child in list_child:
                reprs.append(_flatten(child))
                del node.children[child_id] 

            return reprs

        reprs = _flatten(self.root)
        
        logger.info('Saving PST to {} / {} nodes.'.format(output_path, self.num_nodes))

        f = open(output_path, 'w')
        cPickle.dump(reprs, f)
        cPickle.dump(self.query_dict, f)
        f.close()

    def get_count(self):
        def _count(node):
            count = 0

            list_child = node.children.values()
            for child in list_child:
                count += _count(child)
            return count + len(node.children)
        return _count(self.root)
    
    def delete_children(self, node, to_delete):
        del node.children[to_delete.node_id]

    def prune(self, epsilon=0.05):
        smoothing = 1.0/len(self.query_dict)
        def _prune(node):
            if len(node.probs) > 0:
                list_nodes = node.children.values()                
                kl_nodes = [(x, kl_divergence(node.probs, x.probs, smoothing)) for x in list_nodes]
                for kl_node, kl in kl_nodes:
                    if kl < epsilon:
                        self.delete_children(node, kl_node)
            
            list_nodes = node.children.values()
            for child in list_nodes:
                _prune(child)

        _prune(self.root)
        self.num_nodes = self.get_count()

    def normalize(self):
        def _normalize(node):
            norm = sum(node.probs.values())
            node.probs = dict([(x, v/norm) for x, v in node.probs.items()])
            for child in self.children.values():
                _normalize(child)
        _normalize(node)
    
    def _find(self, suffix):
        past_node = self.root

        for query in reversed(suffix):
            next_node = self.get_child(past_node, query)
            if next_node:
                past_node = next_node
            else:
                return past_node, query, False
        return past_node, None, True 

    def get_child(self, node, query_id): 
        return node.children.get(query_id, None)

    def add_child(self, node, new_node):
        assert new_node.node_id not in node.children
        node.children[new_node.node_id] = new_node
        return new_node

    def add_session(self, session):
        def _update_prob(node, query_id):
            node.probs[query_id] = node.probs.get(query_id, 0.) + 1.

        # Check if the root has a node with that name
        len_session = len(session)
        if len_session < 2:
            return
        
        for x in session:
            if x not in self.query_dict:
                self.query_dict[x] = len(self.query_dict) 
        session = [self.query_dict[x] for x in session]
       
        for x in range(len_session - 1):
            tgt_indx = len_session - x - 1

            for c in range(self.D):
                ctx_indx = tgt_indx - c - 1
                if ctx_indx < 0:
                    break
                
                suffix = session[ctx_indx:tgt_indx]
                tgt = session[tgt_indx]

                assert len(suffix) > 0
                suffix_node, last_query, found = self._find(suffix)    

                if not found:
                    new_node = Node(last_query)
                    suffix_node = self.add_child(suffix_node, new_node)
                    self.num_nodes += 1

                _update_prob(suffix_node, tgt)


