#!/bin/bash
# Run program
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,allow_gc=True,scan.allow_gc=False,nvcc.flags=-use_fast_math python $*
