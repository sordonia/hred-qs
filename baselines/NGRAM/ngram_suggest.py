
from Common import pstfast
import argparse
import cPickle
import operator
import os

def suggest(ctx_files, model_file):
    pstree = pstfast.PSTInfer(exact_match=True)
    pstree.load(model_file)
    
    for ctxf in ctx_files:
        output_file = open(ctxf + "_NGRAM.gen", "w")
        f = open(ctxf, 'r')
        for line in f:
            suggestions = pstree.suggest(line.strip().split('\t'))
            print >> output_file, '\t'.join(suggestions['suggestions'])
        f.close()
        output_file.close()
