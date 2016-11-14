
import os
from Common import pstfast
import argparse
import cPickle
import operator

def suggest(ctx_files, model_file, to_file=1):
    freq_dict = cPickle.load(open(model_file)) 
    most_common = map(lambda x : x[0], freq_dict.most_common()[:10])
     
    for ctxf in ctx_files:
        output_file = open(ctxf + "_FREQ.gen", "w")
        
        f = open(ctxf, 'r')
        for line in open(ctxf):
            print >> output_file, '\t'.join(most_common)
        f.close()

        output_file.close()
