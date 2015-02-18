import pprint
import sys
import cPickle
import numpy

print cPickle.load(open(sys.argv[1], 'r'))[sys.argv[2]]
