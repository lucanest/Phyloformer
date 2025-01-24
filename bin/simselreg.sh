#!/bin/sh

function usage {
  echo ""
  echo "Usage: $0 <TREEDIR> <ALNDIR> <LENGTH> <PASTEK>"
  echo ""
  echo "Simulate alignments from a set of trees using the SelReg model"
  echo ""
  echo "ARGUMENTS:"
  echo "   TREEDIR : Directory containing trees (.nwk) to simulate alignments from"
  echo "   ALNDIR  : Directory in which the alignments will be written"
  echo "   LENGTH  : Total length in AAs of the aligned sequences"
  echo "   PASTEK  : Path to pastek binary"
  exit 1
}

# Check that we have the correct number of arguments
if [ $# -ne 4 ]; then
  usage
fi

INDIR=$(realpath $1)
OUTDIR=$2
LENGTH=$3
PASTEK=$(realpath $4)

# Create output directory if needed
mkdir -p "$OUTDIR"
OUTDIR=$(realpath "$OUTDIR")

echo "Simulating alignments of length $LENGTH for trees in $INDIR into $OUTDIR"

# Go to tree dir
cd "$INDIR"

for treefile in *nwk; do
  $PASTEK multiselreg \
    --alignment-output="$OUTDIR/${treefile/.nwk/}.fasta" \
    --nsites=$LENGTH \
    --seed=42 \
    --selreg-output=regimes.txt \
    --selreg-weights=25,25,25,25 \
    --Ne 0.5 \
    --tree="$PWD/$treefile"
done

# Go back to calling directory
cd -


