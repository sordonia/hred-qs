#!/bin/bash
# Length features
PREF=$1
shift
for i in $*; do
  echo "generating for $i"
  echo "generating LEN ..."
  python main.py SCORE --model LEN . $i/ctx $i/rnk
  # Nsim features
  echo "generating NSIM ..."
  python main.py SCORE --model NSIM . $i/ctx $i/rnk
  # LEV features
  echo "generating LEV ..."
  python main.py SCORE --model LEV . $i/ctx $i/rnk
  # QF features
  echo "generating QF ..."
  python main.py SCORE --model QF "$PREF"_QF.mdl $i/ctx $i/rnk
  # VMM features
  echo "generating VMM ..."
  python main.py SCORE --model VMM "$PREF"_VMM.mdl $i/ctx $i/rnk --no-normalize --fallback 2> /dev/null
  # ADJ features
  echo "generating ADJ ..."
  python main.py SCORE --model ADJ "$PREF"_ADJ.mdl $i/ctx $i/rnk --no-normalize --fallback
done
