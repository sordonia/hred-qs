import os
import sys
import argparse
import cPickle
import operator

def main(model_file):
    query_freq = cPickle.load(open(model_file))
    queries = sorted(query_freq.items(), key=lambda x:x[1], reverse=True) 
    print '\n'.join([x[0] for x in queries[101:10000]])

main(sys.argv[1])
