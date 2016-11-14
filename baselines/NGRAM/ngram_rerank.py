import os
import argparse
import cPickle
import operator
import itertools
from Common.psteff import * 

def rerank(model_file, ctx_file, rnk_file):
    pstree = PSTInfer()
    pstree.load(model_file)
     
    output_file = open(rnk_file + "_NGRAM.gen", "w")
     
    coverage = 0
    for num_line, (ctx_line, rnk_line) in \
            enumerate(itertools.izip(open(ctx_file), open(rnk_file))):
        suffix = ctx_line.strip().split('\t')
        candidates = rnk_line.strip().split('\t')
         
        found, not_found = pstree.rerank(suffix, candidates, exact_match=True)
        coverage += len(found) != 0
         
        print >> output_file, '\t'.join(found + not_found)
     
    output_file.close()
    print 'Coverage {}/{}'.format(coverage, num_line+1)
