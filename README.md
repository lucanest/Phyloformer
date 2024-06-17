<p align="center">
  <img src="https://github.com/lucanest/Phyloformer/blob/main/figures/phyloformer_color.png?raw=true">
</p>

# Phyloformer: towards fast, accurate and versatile phylogenetic reconstruction with deep neural networks
- Luca Nesterenko
- Luc Blassel
- Philippe Veber
- Bastien Boussau
- Laurent Jacob

This repository contains the scripts for [the paper]():

```bibtex
@article{Nesterenko2024phyloformer,
  author={Nesterenko Luca, Luc Blassel, Philippe Veber, Boussau Bastien, Jacob Laurent},
  title={Phyloformer: towards fast, accurate and versatile phylogenetic reconstruction with deep neural networks},
  doi=...,
  year={2024},
  journal={bioRxiv}
}
```

![](./figures/sketch.png)


Phyloformer is a fast deep neural network-based method to infer evolutionary distance from a multiple sequence alignment. 
It can be used to infer alignments under a selection of evolutionary models: LG+GC, LG+GC with indels, CherryML co-evolution model and SelReg with 
selection. 

## Running Phyloformer

### Installing dependencies

The easiest way to install the software is by creating a virutal environment using conda/mamba and then installing this module locally:

```
# Install mamba if you want to use it instead of conda
conda install -n base -c conda-forge mamba

# Clone the phyloformer repo
git clone https://github.com/lucanest/Phyloformer.git && cd Phyloformer

# Create the virtual env and install the phyloformer package inside
conda create -n phylo python=3.9 -c defaults && conda activate phylo
pip install -r requirements.txt
```

Some pre-built binaries are included in this repo both for [linux](./bin/bin_linux/) and [macos](./bin/bin_macos/), these include:
- [`IQTree`](http://www.iqtree.org): for inferring maximum likelihood (ML) trees and simulating alignments *(For the alignment simulation to work you should use IQTree v2.0.0)*
- [`FastTree`](http://www.microbesonline.org/fasttree/): for inferring ML-like trees
- [`FastME`](https://gite.lirmm.fr/atgc/FastME): for inferring trees from distance matrics (such as the ones produced by phyloformer)
- [`goalign`](https://github.com/evolbioinfo/goalign): for manipulating alignments
- [`phylotree`](https://github.com/lucblassel/phylotree-rs): for manipulating newick formatted phylogenetic trees
- [`phylocompare`](https://github.com/lucblassel/phylocompare): for batch comparison of newick formatted phylogenetic trees

If any of these executables do not run on your platform you can find more
information as well as builds and buil-instruction in the links to each tool's
repository.


### Using pre-trained models
All the named phyloformer models in the manuscript are given in the [`models`](./models/) directory:
- `PF_Base` trained with an MAE loss on LG+GC data
- `PF` fine-tuned from PF_Base with an MRE loss on LG+GC data
- `PF_Indel` fine-tuned from PF_Base with an MAE loss on LG+GC+Indels data
- `PF_Cherry` fine-tuned from PF_Base with an MAE loss on CherryML data
- `PF_SelReg` fine-tuned from PF_Base with an MAE loss on SelReg data

Use the [`infer_alns.py`](./infer_alns.py) script to infer some distance matrices from alignments using a trained Phyloformer model

Let's use the small test set given along with this repo to test out Phloformer. 
```shell
# First make sure you are in the repo and have the correct conda env
cd Phyloformer && conda activate phylo

# Infer distance matrices using the LG+GC PF model 
# (This will automatically use a CUDA GPU if available, otherwise it will use the CPU)
python infer_alns.py -o data/testdata/pf_matrices data/testdata/msas

# Infer trees with FastME
mkdir data/testdata/pf_trees
for file in data/testdata/pf_matrices/*; do
  # Get file stem
  base="${file##*/}"
  stem="${base%%.*}"

  # Infer trees
  ./bin/bin_linux/fastme -i "${file}" -o "data/testdata/pf_trees/${stem}.nwk" --nni --spr
done

# Compare trees 
phylocompare -t -n -o data/cmp data/testdata/trees data/testdata/pf_trees

# Check the average normalized KF score
TODO...
```


### Simulating data
Simulate trees with [`simulate_trees.py`](./simulate_trees.py), if you want to simulate LG+GC alignments use [`alisim.py`](./alisim.py). 
If you want to use Cherry to simulate alignments use [`TODO`](todo), for SelReg use [`TODO`](todo).

Let us simulate a small testing set with different tree sizes:

```shell
# Create output directory
mkdir data/test_set

# Simulate 20 trees for each number of tips from 10 to 80 with a step size of 10
for i in $(seq 10 10 80); do
    python simulate_trees.py --ntips "$i" --ntrees 20 --output data/test_set/trees --type birth-death
done

# Simulate 250-AA long alignments using LG+GC from the simulated trees
# here we specify the iqtree binary given in this repo and allow duplicate sequences 
# in the MSAs we get
python alisim.py \
    --outdir data/test_set/alignments \
    --substitution LG \
    --gamma GC \
    --iqtree ./bin/bin_linux/iqtree_2.2.0 \
    --length 250 \
    --allow-duplicates \
    --max-attempts 1 \
    data/test_set/trees
```

### Training a Phyloformer model
Use the [`train_distributed`](./train_distributed.py) script to train or fine-tune a PF model on some data (Need lightning, will work on a SLURM env)
```shell
TODO Add instructions for this
```

### Re-producing figures
Use the [`make_plots`](./make_plots.py) script to reproduce all paper figures. 

```shell
# Download the results (This might take a little time since the file is quite large)
curl <ADDRESS-TO-RESULTS-FILE> --output .

# Extract results file, make sure you extract it to the `data/` directory as that is where the script will look for them
tar xzvf results.tar.gz --directory data/

# Run figure producing script (this should take 5 to 10 minutes)
python make_plots.py
```

