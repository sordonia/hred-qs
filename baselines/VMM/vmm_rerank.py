import os
import argparse
import cPickle
import operator
import itertools
from Common.psteff import *

def rerank(model_file, ctx_file, rnk_file, score=False, no_normalize=False, fallback=False):
    pstree = PSTInfer()
    pstree.load(model_file)
    output_file = open(rnk_file + "_VMM" + (".f" if score else ".gen"), "w")
    begin=True
    for ctx_line, rnk_line in itertools.izip(open(ctx_file), open(rnk_file)):
        suffix = ctx_line.strip().split('\t')
        candidates = rnk_line.strip().split('\t')
        candidates, scores = pstree.rerank(suffix, candidates, no_normalize=no_normalize, fallback=fallback)
        if not score:
            reranked = [x[0] for x in sorted(zip(candidates, scores),
                                             key=operator.itemgetter(1),
                                             reverse=True)]
            print >> output_file, '\t'.join(reranked)
        else:
            if begin:
                print >> output_file, 'VMM'
                begin=False

            for s in scores:
                print >> output_file, s 
    output_file.close()
