from Common import psteff
import sys
import collections
import operator
import cPickle
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def get_query_to_clst(cluster_map):
    q_to_c = {}
    
    # Here we assign a cluster  
    # to each query by a dictionary q_to_c
    for cluster, data in cluster_map.items():
        for query in data['queries']:
            q_to_c[query] = cluster
    
    return q_to_c

class CACBInfer(psteff.PSTInfer):
    def __init__(self):
        psteff.PSTInfer.__init__(self)
    
    def _load_pickle(self, input_file):
        f = open(input_file, 'r')     
        self.data = cPickle.load(f)
        self.query_dict = cPickle.load(f)
        self.cluster_map = cPickle.load(f)
        self.inv_query_dict = sorted(self.query_dict.items(), key=operator.itemgetter(1))
        f.close()
        
        self.query_to_clst = get_query_to_clst(self.cluster_map)

    def suggest(self, prefix):
        # Convert to cluster ids
        concepts = [self.query_to_clst.get(q, q) for q in prefix]
        concepts = [c for n, c in enumerate(concepts) if n == 0 or concepts[n] != concepts[n-1]]
        
        # Suggestion is a list of cluster ids        
        suggestions = pstfast.PSTInfer.suggest(self, concepts)
        sugg = zip(suggestions['suggestions'], suggestions['scores'])
        suggestions['suggestions'] = []
        suggestions['scores'] = []
        suggestions['clusters'] = []
        for clst_sugg, prob in sugg:
            if clst_sugg in self.cluster_map:
                # Check the concept id and get the query with
                # the most clicks
                sugg = self.cluster_map[clst_sugg]['queries'][0]
                score = self.cluster_map[clst_sugg]['ranks'][0]
            else:
                # The concept id is the query
                sugg = clst_sugg
                score = 0
            
            suggestions['clusters'].append(clst_sugg)
            suggestions['suggestions'].append(sugg)
            suggestions['scores'].append(score)
        return suggestions

class CACB(psteff.PST):
    def __init__(self, D=4):
        psteff.PST.__init__(self, D) 
        self.query_to_clst = {}
        self.cluster_map = {}

    def with_cluster(self, cluster_file):
        self.cluster_map = cPickle.load(open(cluster_file, 'r'))
        self.query_to_clst = get_query_to_clst(self.cluster_map)

        logger.info('Loaded {} queries'.format(len(self.query_to_clst)))
        logger.info('Loaded {} clusters'.format(len(self.cluster_map)))
    
    def prune(self, K=5):
        smoothing = 1.0/len(self.query_dict)
        def _prune(node, parent):
            if len(node.probs) > 0:
                total_sequence_freq = sum(node.probs.values())
                if total_sequence_freq < 5:
                    self.delete_children(parent, node)
                else:
                    for child in list_nodes:
                        _prune(child, node)
        _prune(self.root, None)
        self.num_nodes = self.get_count()

    def save(self, output_path):
        def _flatten(node):
            sorted_probs = sorted(node.probs.items(), key=operator.itemgetter(1), reverse=True)[:10]
            reprs = [(node.node_id, sorted_probs)]
             
            list_child = node.children.values()
            for child in list_child:
                reprs.append(_flatten(child))
            return reprs

        reprs = _flatten(self.root)

        logger.info('Saving CACB to {} / {} nodes.'.format(output_path, self.num_nodes))
        
        f = open(output_path, 'w')
        cPickle.dump(reprs, f)
        cPickle.dump(self.query_dict, f)
        cPickle.dump(self.cluster_map, f)
        f.close()
    
    def add_session(self, session):
        prefix = session.strip().split('\t')
        
        coverage = 0
        concepts = []
        for q in prefix:
            if q in self.query_to_clst:
                concepts.append(self.query_to_clst[q])
                coverage += 1
            else:
                concepts.append(q)

        concepts = [c for n, c in enumerate(concepts) if n == 0 or concepts[n] != concepts[n-1]]
        psteff.PST.add_session(self, concepts)
        return coverage
