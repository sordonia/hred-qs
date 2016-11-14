#!/usr/bin/python

from NGRAM import ngram_builder, ngram_suggest, ngram_rerank
from VMM import vmm_builder, vmm_suggest, vmm_rerank
from FREQ import freq_builder, freq_suggest
from CACB import cacb_builder, cacb_suggest, cacb_rerank
from ADJ import adj_builder, adj_suggest, adj_rerank
from QF import qf_builder, qf_rerank
from NSIM import nsim_rerank
from LEN import len_rerank
from LEV import lev_rerank

import logging
import cPickle
import os
import sys
import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='BUILD or SCORE')
    parser.add_argument('input_file', type=str,
                        help='Model file or Session file')
    parser.add_argument('--model', default='', type=str,
                        help='Build NGRAM/QF/CACB/VMM/ADJ')
    parser.add_argument('ext_file', nargs='*')
    parser.add_argument('--no-normalize', action='store_true')
    parser.add_argument('--fallback', action='store_true')
    parser.add_argument('--epsilon', type=float, default=0)

    args = parser.parse_args()
    assert os.path.isfile(args.input_file) or os.path.isdir(args.input_file)
    if args.action == 'BUILD':
      assert args.model != ''

    if args.action == 'BUILD':
        if args.model == 'NGRAM':
            ngram_builder.build(args.input_file)
        elif args.model == 'QF':
            qf_builder.build(args.input_file)
        elif args.model == 'CACB':
            assert args.ext_file != ''
            cacb_builder.build(args.ext_file[0], args.input_file)
        elif args.model == 'VMM':
            vmm_builder.build(args.input_file, args.epsilon)
        elif args.model == 'ADJ':
            adj_builder.build(args.input_file)
        else:
            raise Exception('Model not known!')
    if args.action == 'SCORE':
        if args.model == 'LEN':
            len_rerank.rerank('', *args.ext_file, score=True)
            sys.exit(-1)
        if args.model == 'NSIM':
            nsim_rerank.rerank('', *args.ext_file, score=True)
            sys.exit(-1)
        if args.model == 'LEV':
            lev_rerank.rerank('', *args.ext_file, score=True)
            sys.exit(-1)
        # the following needs a model file specified by input file
        sta = args.input_file.rfind('_')
        assert args.input_file[sta+1:-4] == args.model or args.input_file[sta+1:-5] == args.model
        if args.model == 'CACB':
            cacb_rerank.rerank(args.input_file, *args.ext_file, score=True, fallback=args.fallback)
        elif args.model == 'QF':
            qf_rerank.rerank(args.input_file, *args.ext_file, score=True)
        elif args.model == 'VMM':
            vmm_rerank.rerank(args.input_file, *args.ext_file, score=True, no_normalize=args.no_normalize, fallback=args.fallback)
        elif args.model == 'NGRAM':
            ngram_rerank.rerank(args.input_file, *args.ext_file, score=True)
        elif args.model == 'ADJ':
            adj_rerank.rerank(args.input_file, *args.ext_file, score=True, no_normalize=args.no_normalize, fallback=args.fallback)
        else:
            raise Exception('Model not known!')
