
import os
from Common import pstfast
import argparse
import cPickle
import operator

def suggest(ctx_files, model_file, to_file=1):
    pstree = pstfast.PSTInfer()
    pstree.load(model_file)
    
    for ctxf in ctx_files:
        output_file = open(ctxf + "_ADJ.gen", "w")
        f = open(ctxf, 'r')
        for line in open(ctxf):
            suggestions = pstree.suggest(line.strip().split('\t'))
            print >> output_file, '\t'.join(suggestions['suggestions'])
        f.close()
        output_file.close()
