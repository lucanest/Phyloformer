#!/bin/sh

function usage {
  echo ""
  echo "Usage: $0 <TREEDIR> <ALNDIR> <LENGTH>"
  echo ""
  echo "Simulate alignments from a set of trees using the CherryML model"
  echo ""
  echo "ARGUMENTS:"
  echo "   TREEDIR : Directory containing trees (.nwk|.newick) to simulate alignments from"
  echo "   ALNDIR  : Directory in which the alignments will be written"
  echo "   LENGTH  : Total length in AAs of the aligned sequences"
  echo "               (will result in n/2 pairs of correlated amino acids"
  echo "               for a total length of n)"
  exit 1
}

# Check that we have the correct number of arguments
if [ $# -ne 3 ]; then
  usage
fi

INDIR=$1
OUTDIR=$2
LENGTH=$3
HALF_LENGTH=$((LENGTH / 2))
BINDIR=$(dirname $(realpath $0))

echo "Simulating alignments of length $LENGTH for trees in $INDIR into $OUTDIR"

mkdir -p "$OUTDIR"

python "${BINDIR}/simulateWithCoevolution/src/simulateGillespie.py" \
    --exchangeabilities "${BINDIR}/simulateWithCoevolution/data/coevolution.txt" \
    --eqfreq "${BINDIR}/simulateWithCoevolution/data/coevolution_stationary.txt" \
    --trees "$INDIR" \
    --output "$OUTDIR" \
    --seqlen "$HALF_LENGTH" 
