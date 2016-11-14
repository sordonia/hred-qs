
import argparse
import cPickle
import operator
import os
from Common.psteff import * 

def suggest(ctx_files, model_file):
    pstree = PSTInfer()
    pstree.load(model_file)
 
    for ctxf in ctx_files:
        output_file = open(ctxf + "_VMM.gen", "w")
        f = open(ctxf, 'r')
        for num, line in enumerate(f):
            suggestions = pstree.suggest(line.strip().split('\t'))
            print >> output_file, '\t'.join(suggestions['suggestions'])
            output_file.flush()
        f.close()
        output_file.close()   
