import os
import argparse
import cPickle
import operator
import itertools
from Common.psteff import * 

def rerank(model_file, ctx_file, rnk_file, score=False):
    query_freq = cPickle.load(open(model_file))
     
    begin=True
    output_file = open(rnk_file + "_QF" + (".f" if score else ".gen"), "w")    
    for ctx_line, rnk_line in itertools.izip(open(ctx_file), open(rnk_file)):
        suffix = ctx_line.strip().split('\t')
        candidates = rnk_line.strip().split('\t')
        
        s_score = query_freq.get(suffix[-1], 0.)
        c_scores = [query_freq.get(query, 0.) for query in candidates] 
        if not score:
            raise Exception('Not supported!')
        else:
            if begin:
                print >> output_file, 'cQF', 'sQF'
                begin=False
            for c_score in c_scores:
                print >> output_file, c_score, s_score 
    output_file.close()
