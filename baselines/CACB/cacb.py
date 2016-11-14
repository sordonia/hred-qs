from Common import psteff
import sys
import collections
import operator
import cPickle
import logging
import numpy

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def _prepare_concepts(query_to_clst, session):
    concepts = map(lambda query: query_to_clst.get(query, query), session)
    # Remove same subsequent concepts
    concepts = [c for n, c in enumerate(concepts) \
                    if n == 0 or concepts[n] != concepts[n-1]]
    return concepts

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
    
    def _load_pickle(self, input_handle):
        self.tuple_dict = cPickle.load(input_handle)
        self.query_to_id = cPickle.load(input_handle)
        self.cluster_map = cPickle.load(input_handle)

        logger.info('Creating query to cluster map')
        self.query_to_clst = get_query_to_clst(self.cluster_map)

        logger.info('{} in tuple_dict'.format(len(self.tuple_dict)))

    def suggest(self, prefix):
        concepts = _prepare_concepts(self.query_to_clst, prefix)

        # Suggestion is a list of cluster ids        
        suggestions = psteff.PSTInfer.suggest(self, concepts)
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

    def rerank(self, suffix, candidates, no_normalize=False):
        concepts = _prepare_concepts(self.query_to_clst, suffix) 
         
        probs = []
        for i in range(len(concepts)):
            probs.append(self._find(concepts[i:]))
         
        ids_candidates = map(lambda candidate: self.query_to_id. \
                                get(self.query_to_clst.          \
                                    get(candidate, candidate), -1), candidates)
         
        n_total_queries = len(self.query_to_clst)

        reranked = []
        for (id_candidate, candidate) in zip(ids_candidates, \
                                             candidates):
            candidate_prob = 0. 
            for prob in probs:
                if id_candidate in prob['probs']:
                    if no_normalize:
                        candidate_prob = prob['probs'][id_candidate]
                    else:
                        freq = prob['probs'][id_candidate]
                        total_freq = sum(prob['probs'].values())

                        n_remaining_queries = (n_total_queries - len(prob['probs']))
                        assert n_remaining_queries >= 0 

                        candidate_prob = float(freq)/total_freq
                        candidate_prob = candidate_prob/(candidate_prob \
                                                         + float(n_remaining_queries)/n_total_queries)
                        candidate_prob = candidate_prob 
                    break
            reranked.append((candidate, candidate_prob)) 
        return zip(*reranked) 

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
    
    def save(self, output_path):
        logger.info('Saving CACB to {} / {} nodes.'.format(output_path, self.num_nodes))
         
        # Save the normalized format
        # if not self.normalized and not no_normalize:
        #    logger.info('Normalizing PST')
        #    for key, count in self.tuple_dict.iteritems():
        #        self.tuple_dict[key] = float(count)/self.norm_dict.get(key[:-1])        
        # self.norm_dict.clear()

        f = open(output_path, 'w')
        cPickle.dump(self.tuple_dict, f)
        cPickle.dump(self.query_dict, f)
        cPickle.dump(self.cluster_map, f)
        f.close()
    
    def add_session(self, session):
        coverage = sum([1 for query in session if query in self.query_to_clst])
        concepts = _prepare_concepts(self.query_to_clst, session)
        psteff.PST.add_session(self, concepts)
        return coverage
