import numpy

def count(suggestions):
    return sum([len(s) for s in suggestions])

def count_letter_ngram(sentence, n=3):
    if len(sentence) < n:
        return set(sentence)
    local_counts = set()
    for k in range(len(sentence.strip()) - n + 1): 
        local_counts.add(sentence[k:k+n])
    return local_counts

def _get_ranks(suggestions, targets):
    ranks = []
    for tgt, sugg in zip(targets, suggestions):
        ranks.append([])
        for pos, t in enumerate(tgt):
            if t in set(sugg):
                # Assign relevance score to that position 
                pos_in_sugg = sugg.index(t) + 1
            else:
                pos_in_sugg = numpy.inf
            if pos_in_sugg == numpy.inf:
                pos_in_sugg = len(sugg)
                # print tgt, sugg
            assert pos_in_sugg != numpy.inf
            ranks[-1].append(((5.0-pos), pos_in_sugg))
    assert len(ranks) == len(suggestions)
    return ranks

class Jacc(object):
    def __init__(self):
        self.name = "Jaccard"

    def apply(self, suggestions, targets):
        jaccard = []
        for tgt, sugg in zip(targets, suggestions):
            if not len(sugg):
                continue

            j = 0.
            words_s = set(sugg[0].split()) 
            for t in tgt:
                words_t = set(t.split())
                union = words_s | words_t
                inter = words_s & words_t
                j += float(len(inter))/len(union)

            jaccard.append(j/len(tgt))
        return numpy.mean(jaccard)

class Count(object):
    def __init__(self):
        self.name = 'Count'

    def apply(self, suggestions, targets):
        return len(targets)

class Jacc3(object):
    def __init__(self):
        self.name = 'Jaccard3'

    def apply(self, suggestions, targets): 
        def _jac(s_n, t_n):
            return float(len(s_n & t_n))/len(s_n | t_n)

        jaccard = [] 
        for tgt, sugg in zip(targets, suggestions):
            t_n = count_letter_ngram(tgt[0])
            if len(sugg) == 0:
                continue
            jaccard.append(_jac(count_letter_ngram(sugg[0]), t_n)) 
        return numpy.sum(jaccard)/len(targets)

class MRR(object):
    def __init__(self):
        self.name = 'MRR'

    def apply(self, suggestions, targets):
        ranks = _get_ranks(suggestions, targets)
        num_positive = len(suggestions)

        rec_rank = 0 
        # MRR of first relevant
        for rank in ranks:
            first_ranked = sorted(rank, key=lambda x: x[1])[0]
            rec_rank += 1.0/first_ranked[1]
        return rec_rank/num_positive

class SR(object):
    def __init__(self, K):
        self.K = K
        self.name = 'SR@{}'.format(K)

    def apply(self, suggestions, targets):
        def _sr_at_k(ranks, k):
            found_queries = 0
            for rank in ranks:
                found_queries += sum([1.0 for (r, p) in rank if p <= k])
            return found_queries
         
        ranks = _get_ranks(suggestions, targets)
        num_targets = count(targets)
        
        sr_at_k = float(_sr_at_k(ranks, self.K))/num_targets
        return sr_at_k

class NDCG(object):
    def __init__(self, K):
        self.K = K
        self.name = 'nDCG@{}'.format(K)

    def apply(self, suggestions, targets):
        def _ndcg_at_k(rank, k):
            dcg = 0.
            for rel, pos in rank:
                if pos <= k:
                    dcg += float(2**rel - 1)/numpy.log(pos+1) 
             
            odcg = sorted(rank, key=lambda x: x[0], reverse=True)
            odcg = sum([ (2**(r)-1)/numpy.log(s+2) for s, (r, _) in enumerate(odcg[:k]) ])
            return dcg/odcg

        ranks = _get_ranks(suggestions, targets) 
        ndcg_k = numpy.mean(map(lambda x: _ndcg_at_k(x, self.K), ranks))
        return ndcg_k

class PREC(object):
    def __init__(self, K):
        self.K = K
        self.name = 'P@{}'.format(K)

    def apply(self, suggestions, targets):
        def _p_at_k(rank, k):
            prec = 0.
            for rel, pos in rank:
                if pos <= k and pos != -1:
                    prec += 1.0
            return prec/k
         
        ranks = _get_ranks(suggestions, targets) 
        num_positives = len(suggestions)

        p_k = sum(map(lambda x: _p_at_k(x, self.K), ranks))/num_positive
        return p_k 

