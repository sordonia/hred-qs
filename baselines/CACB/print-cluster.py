

import cPickle
import sys
import operator
import itertools

clusters = cPickle.load(open(sys.argv[1]))
by_length = sorted(clusters.items(), key=lambda (cid, data): len(data['queries']), reverse=True)

print '>> Num clusters {}'.format(len(clusters))
print by_length[0]
