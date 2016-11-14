
import pstfast
import argparse
import cPickle
import operator
import sys

if __name__ == "__main__":
    pstree = pstfast.PST(4)
    pstree.load(sys.argv[1])
    print pstree
