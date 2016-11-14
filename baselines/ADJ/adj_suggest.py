
import os
import argparse
import cPickle
import operator
from Common import psteff

def suggest(ctx_files, model_file, to_file=1):
    pstree = psteff.PSTInfer()
    pstree.load(model_file)
     
    for ctxf in ctx_files:
        output_file = open(ctxf + "_ADJ.gen", "w")
        f = open(ctxf, 'r')
        for line in open(ctxf):
            suffix = line.strip().split('\t')
            suggestions = pstree.suggest(suffix)
            print >> output_file, '\t'.join(suggestions['suggestions'])
        f.close()
        output_file.close()
