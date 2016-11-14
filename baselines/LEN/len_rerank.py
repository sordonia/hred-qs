import os
import argparse
import cPickle
import operator
import itertools

def rerank(model_file, ctx_file, rnk_file, score=False):
    output_wfile = open(rnk_file + "_LEN" + (".f" if score else ".gen"), "w")

    begin = True
    for ctx_line, rnk_line in itertools.izip(open(ctx_file), open(rnk_file)):
        suffix = ctx_line.strip().split('\t')
        candidates = rnk_line.strip().split('\t')

        # Character length
        cscores = map(len, candidates)
        # Word length
        wscores = map(len, [x.split() for x in candidates])
        if not score:
            raise Exception('Not supported!')
        else:
            if begin:
                print >> output_wfile, 'cLEN wLEN allLEN'
                begin=False

            for wscore, cscore in zip(wscores, cscores):
                print >> output_wfile, cscore, wscore, len(suffix)
    output_wfile.close()
