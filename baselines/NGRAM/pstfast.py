import cPickle
import numpy
import os
import sys
import argparse
import collections
import operator
from collections import OrderedDict

def kl_divergence(a, b, smoothing):
    norm_a = sum(a.values())
    norm_b = sum(b.values())
     
    # Approximate KL (i.e. smooth while computing)
    kl = 0.
    for x, v in a.items():
        kl += v/norm_a * (numpy.log(v/norm_a) - numpy.log(b.get(x, smoothing)/norm_b)) 
    assert kl >= 0
    return kl

class Node(object):
    def __init__(self, query_id=-1):
        # Query id
        self.node_id = query_id
        self.probs = {}
        self.children = {}

class PSTInfer(object):
    def __init__(self):
        self.query_dict = {}
        self.inv_query_dict = []    
    
    def _load_pickle(self, input_file):
        f = open(input_file, 'r')     
        self.data = cPickle.load(f)
        self.query_dict = cPickle.load(f)
        f.close()
 
    def load(self, input_file):
        self._load_pickle(input_file)
        self.inv_query_dict = sorted(self.query_dict.items(), key=operator.itemgetter(1))
        print 'Finished loading inference engine'

    def _find(self, suffix):
        def _get_child(level, child):
            for x in level[1:]:
                if x[0][0] == child:
                    return x
            return None
        
        found = False
        past_node = self.data
        for query in reversed(suffix):
            next_node = _get_child(past_node, query)
            if next_node:
                past_node = next_node
            else:
                return past_node[0], False        
        # If the past node id is not the root 
        if past_node != self.data:
            found = True
        return past_node[0], found

    def suggest(self, suffix):
        suffix = [self.query_dict[x] for x in suffix if x in self.query_dict]
        (node_id, scores), found = self._find(suffix)
        
        data = {'last_node_id' : node_id, 'found' : found, 'suggestions' : [], 'scores' : []}
        
        if node_id == -1 or found == False:
            return data 
         
        sugg_score = [(self.inv_query_dict[child_id][0], p) for child_id, p in scores]
        sugg, score = zip(*sugg_score)
        data['suggestions'] = sugg
        data['scores'] = score
        return data

class PST(object):
    def __init__(self, D=2):
        self.root = Node(-1)

        self.query_dict = {}
        self.inv_query_dict = []
        
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
        def _flatten(node):
            # Save top-10 probabilities in increasing order
            sorted_probs = sorted(node.probs.items(), key=operator.itemgetter(1), reverse=True)[:10]
            reprs = [(node.node_id, sorted_probs)]

            list_child = node.children.values()
            for child in list_child:
                reprs.append(_flatten(child))
            return reprs

        reprs = _flatten(self.root) 
        print 'Saving PST to {} / {} nodes.'.format(output_path, self.num_nodes)
        
        f = open(output_path, 'w')
        cPickle.dump(reprs, f)
        cPickle.dump(self.query_dict, f)
        print 'Done.'
    
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


