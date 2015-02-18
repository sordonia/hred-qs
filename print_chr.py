import pprint
import sys
import cPickle
import numpy

print "{}".format(pprint.pformat(cPickle.load(open(sys.argv[1] + "_state.pkl", 'r'))))
print "{}".format(dict(numpy.load(open(sys.argv[1] + "_timing.npz", "r"))))
