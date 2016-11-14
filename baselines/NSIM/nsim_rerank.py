import os
import argparse
import cPickle
import operator
import itertools
from Common import evaluation

def jaccard(ng1, ng2):
    return float(len(ng1 & ng2))/len(ng1 | ng2)

def matches(ng1, ng2):
    return len(ng1 & ng2)

def rerank(model_file, ctx_file, rnk_file, score=False):
    output_sfile = open(rnk_file + "_nSIM" + (".f" if score else ".gen"), "w")

    begin = True
    for ctx_line, rnk_line in itertools.izip(open(ctx_file), open(rnk_file)):
        suffix = ctx_line.strip().split('\t')
        candidates = rnk_line.strip().split('\t')
         
        similarities = []

        for cnd in candidates:
            cnd_ngram = evaluation.count_letter_ngram(cnd)
            assert len(cnd_ngram) != 0
             
            sims = (map(lambda y: \
                        matches(cnd_ngram, evaluation.count_letter_ngram(y)), \
                        reversed(suffix[:10])))
            while len(sims) < 10:
                sims.append(0)
            similarities.append(sims)

        if not score:
            raise Exception('Not supported!')
        else:
            if begin:
                print >> output_sfile, ' '.join(map(lambda x: '%dSIM' % x, xrange(1, 11)))
                begin=False
            
            for similarity in similarities:
                print >> output_sfile, ' '.join(map(lambda x: '%f' % x, similarity))

    output_sfile.close()
