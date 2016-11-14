
import cacb
import argparse
import cPickle
import operator
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('cacb', type=str, help='Cacb file')
    parser.add_argument('ctx_file', type=str, help='Query contexts')
    args = parser.parse_args()
     
    assert os.path.isfile(args.cacb) and os.path.isfile(args.ctx_file)
     
    cacbt = cacb.CACBInfer()
    cacbt.load(args.cacb)
     
    for line in open(args.ctx_file):
        suggestions = cacbt.suggest(line.strip().split('\t'))
        print suggestions
