import os
import argparse
import cPickle
import operator
import itertools
from Common import evaluation

def lev_dist(first, second):
    """Find the Levenshtein distance between two strings."""
    if len(first) > len(second):
        first, second = second, first
    if len(second) == 0:
        return len(first)
    first_length = len(first) + 1
    second_length = len(second) + 1
    distance_matrix = [[0] * second_length for x in range(first_length)]
    for i in range(first_length):
       distance_matrix[i][0] = i
    for j in range(second_length):
       distance_matrix[0][j]=j
    for i in xrange(1, first_length):
        for j in range(1, second_length):
            deletion = distance_matrix[i-1][j] + 1
            insertion = distance_matrix[i][j-1] + 1
            substitution = distance_matrix[i-1][j-1]
            if first[i-1] != second[j-1]:
                substitution += 1
            distance_matrix[i][j] = min(insertion, deletion, substitution)
    return distance_matrix[first_length-1][second_length-1]

def rerank(model_file, ctx_file, rnk_file, score=False):
    output_sfile = open(rnk_file + "_nLEV" + (".f" if score else ".gen"), "w")

    begin = True
    for ctx_line, rnk_line in itertools.izip(open(ctx_file), open(rnk_file)):
        suffix = ctx_line.strip().split('\t')
        candidates = rnk_line.strip().split('\t')
         
        sims = []
        for cnd in candidates:
            s = (map(lambda y: \
                        lev_dist(cnd, y), \
                        reversed(suffix[:10])))
            
            avg_lev = float(sum(s))/len(suffix)
            last_lev = s[0]
            sims.append((last_lev, avg_lev))

        if not score:
            raise Exception('Not supported!')
        else:
            if begin:
                print >> output_sfile, 'lastLEV', 'avgLEV' 
                begin=False
            
            for l, a in sims: 
                print >> output_sfile, l, a 
        
        output_sfile.flush()
    output_sfile.close()
