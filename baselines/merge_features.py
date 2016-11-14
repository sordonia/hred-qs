#!/usr/bin/python
import logging
import numpy
import cPickle
import os
import sys
import collections
import argparse

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help='input dir')
    parser.add_argument('--feature-file', type=str, help='feature file', default='')
    args = parser.parse_args()

    assert os.path.isfile(args.input_dir) or os.path.isdir(args.input_dir)

    f_name_to_id = collections.OrderedDict()
    f_name_to_scores = collections.OrderedDict()

    all_candidates = [line.strip().split('\t') for line in open(args.input_dir + '/rnk').readlines()]
    all_targets = [line.strip() for line in open(args.input_dir + '/tgt').readlines()]

    output_feature_dict = True
    if args.feature_file != '':
        output_feature_dict = False
        f_file_lines = [line.strip().split(' # ') for line in open(args.feature_file)]
        for f_id, f_name in f_file_lines :
            logging.info('Adding {} - {}'.format(f_id, f_name))
            f_name_to_id[f_name] = int(f_id)

    f_files = []
    for fn in os.listdir(args.input_dir):
        fn = args.input_dir + '/' + fn
        if os.path.isfile(fn) and fn[-2:] == '.f':
            f_files.append(fn)

    f_files = sorted(f_files)
    for f_file in f_files:
        # Handle multi-column files
        f_records = open(f_file, 'r').readlines()
        f_all_names = f_records[0].strip().split()
        f_all_scores = [map(float, x.strip().split()) for x in f_records[1:]]

        # Transpose scores
        f_all_scores = list(numpy.array(f_all_scores).T)

        for f_name, f_scores in zip(f_all_names, f_all_scores):
            assert f_name not in f_name_to_scores
            if not output_feature_dict:
                assert f_name in f_name_to_id
            else:
                assert f_name not in f_name_to_id
                f_name_to_id[f_name] = len(f_name_to_id) + 1
            f_name_to_scores[f_name] = f_scores
            logging.info('Feature {} -> {}, {}'.format(f_name, f_name_to_id[f_name], len(f_scores)))

    output_feats_file = open(args.input_dir + '/rnk.feats', 'w')
    for f_name, f_id in f_name_to_id.items():
        print >> output_feats_file, f_id, '#', f_name
    output_feats_file.close()
    output_file = open(args.input_dir + '/rnk.mart', 'w')

    total = 0
    for qid, (target, candidates) in enumerate(zip(all_targets, all_candidates)):
        if qid % 1000 == 0:
            logging.info('Doing {}'.format(total))
        for candidate in candidates:
            print_string = '%d' % (candidate==target) \
                + ' qid:%d' % (qid+1) \
                + ''.join([' %d:%.8f' % (f_name_to_id[f_name],f_name_to_scores[f_name][total]) for f_name in f_name_to_id.keys()])
            print >> output_file, print_string
            total += 1
    output_file.close()

if __name__ == "__main__":
    main()
