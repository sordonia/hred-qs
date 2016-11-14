
import cacb
import argparse
import cPickle
import operator
import os

def suggest(ctx_files, model_file):    
    cacbt = cacb.CACBInfer()
    cacbt.load(model_file)
     
    for ctxf in ctx_files:
        output_file = open(ctxf + "_CACB.gen", "w")
        f = open(ctxf, 'r')
        for line in f: 
            suggestions = cacbt.suggest(line.strip().split('\t'))
            print >> output_file, '\t'.join(suggestions['suggestions'])
        f.close()
        output_file.close()
