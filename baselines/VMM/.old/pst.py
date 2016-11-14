import cPickle
import numpy
import os
import sys
import argparse
import collections
from collections import OrderedDict

def kl_divergence(a, b, smoothing):
    norm_a = sum(a.values())
    norm_b = sum(b.values())
    
    # Approximate KL (i.e. smooth while computing)
    kl = 0.
    for x, v in a.items():
        kl += v/norm_a * (numpy.log(v/norm_a) - numpy.log(b.get(x, smoothing)/norm_b)) 
    return kl

class Node(object):
    def __init__(self, query_id=-1):
        # Query id
        self.node_id = query_id
        self.probs = {}
        # Sorted array of children
        self.children = {}
    
    def update_prob(self, query_id):
        self.probs[query_id] = self.probs.get(query_id, 0.) + 1.
    
    def prune(self, epsilon, smoothing):
        if len(self.probs) > 0:
            list_nodes = self.children.items()
            kl_nodes = [(xid, kl_divergence(self.probs, x.probs, smoothing)) for xid, x in list_nodes]
            
            for xid, kl in kl_nodes:
                if kl < epsilon:
                    del self.children[xid]
        
        for xid, node in self.children.items():
            node.prune(epsilon, smoothing)

    def normalize(self):
        norm = sum(self.probs.values())
        self.probs = dict([(x, v/norm) for x, v in self.probs.items()])
        for node_id, node in self.children.items():
            node.normalize()

    def add_child(self, query_id):
        new_node = Node(query_id)
        self.children[query_id] = new_node
        return new_node
    
    def to_string(self, space=0):
        message = '\t' * space + '@@NODE: %s - @@PROBS: %s' % (self.node_id, self.probs.items())
        for node_id, node in self.children.items():
            message += '\n%s' % node.to_string(space + 1)
        return message

    def get_children(self):
        return self.children.items()

    def get_child(self, node_id):
        if node_id in self.children:
            return self.children[node_id]
        return None

    def get_count(self):
        count = 0
        for x, v in self.children.items():
            count += v.get_count()
        return count + len(self.children)

class PST(object):
    def __init__(self, D=2):
        self.root = Node()
        self.nodes = [self.root]
        self.D = D
        self.num_nodes = 1

    def __str__(self):
        return self.root.to_string() 
    
    def get_children(self):
        def _get_children(node):
            nodes = [node]
            for x, v in node.children.items():
                nodes += _get_children(v)
            return nodes
         
        return _get_children(self.root)

    def save(self, output_path):
        self.nodes = self.get_children()
        print 'Saving PST to {} / {} nodes.'.format(output_path, len(self.nodes)) 
        cPickle.dump(self.nodes, open(output_path, 'w'))
        print 'Done.'
    
    def load(self, input_file):
        self.nodes = cPickle.load(open(input_file, 'r'))
        self.root = self.nodes[0]
        self.num_nodes = len(self.nodes)

    def prune(self, epsilon=0.05, smoothing=1e-5):
        self.root.prune(epsilon, smoothing)
        num_nodes = self.root.get_count()

    def normalize(self):
        self.root.normalize()

    def find(self, suffix):
        past_node = self.root

        for query in reversed(suffix):
            next_node = past_node.get_child(query)
            if next_node:
                past_node = next_node
            else:
                return past_node, query, False
        return past_node, None, True 

    def add_session(self, session):
        # Check if the root has a node with that name
        len_session = len(session)
        if len_session < 2:
            return
        
        for x in range(len_session - 1):
            tgt_indx = len_session - x - 1

            for c in range(self.D):
                ctx_indx = tgt_indx - c - 1
                if ctx_indx < 0:
                    break
                
                suffix = session[ctx_indx:tgt_indx]
                tgt = session[tgt_indx]

                assert len(suffix) > 0
                suffix_node, last_query, found = self.find(suffix)    

                if not found:
                    suffix_node = suffix_node.add_child(last_query)
                    self.num_nodes += 1

                suffix_node.update_prob(tgt)


