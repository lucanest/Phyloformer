<p align="center">
  <img src="https://github.com/lucanest/Phyloformer/blob/main/figures/phyloformer_color.png?raw=true">
</p>

# Phyloformer: Towards fast and accurate Phylogeny estimation with self-attention Networks

- Luca Nesterenko
- Bastien Boussau
- Laurent Jacob

This repository contains the scripts for [the paper](https://www.biorxiv.org/content/10.1101/2022.06.24.496975v1):

```bibtex
@article{Nesterenko2022phyloformer,
  author={Nesterenko Luca, Boussau Bastien, Jacob Laurent},
  title={Phyloformer: towards fast and accurate phylogeny estimation with self-attention networks},
  year={2022},
  doi={https://doi.org/10.1101/2022.06.24.496975},
  url={https://www.biorxiv.org/content/10.1101/2022.06.24.496975v1},
  journal={bioRxiv}
}
```

![](https://github.com/lucanest/Phyloformer/blob/main/figures/sketch.png?raw=true)

## Install

The easiest way to install the software is by creating a virutal environment using conda/mamba and then installing this module locally:

```
# Install mamba if you want to use it instead of conda
conda install -n base -c conda-forge mamba

# Clone the phyloformer repo
git clone https://github.com/lucanest/Phyloformer.git && cd Phyloformer

# Create the virtual env and install the phyloformer package inside
conda create -n phylo python=3.9
conda activate phylo
pip install .
```

### Test run

To check that the installation is successful one can run which will infer trees from the test alignments, using a pre-trained model, and save them in that directory

```
predict testdata/alignments
```

and then compare the true trees and their corresponding predictions:

```
evaluate --true testdata/trees --predictions testdata/alignments
```

the printed mean normalized Robinson-Fould distance should be equal to 0.063.

## Using Phyloformer

You can use `phyloformer` as a [library](#todo) in you own python scripts, or through the command line tools that are made installed with the package when following [these steps](#install).

### Command line usage

A description of how and why to use the different scripts is given below. A [quick reference](./cli_reference.md) is also available.

#### Inferring trees

If you just want to infer phylogenetic trees from alignments using a pre-trained instance of `phyloformer` you can use the `predict` script which should be installed in you virtual environment.

```
predict /path/to/alignments/directory
```

providing as argument a directory containing multiple sequence protein alignments in .fasta format,
the program will then predict a phylogenetic tree for each alignment and write them in Newick format in the same directory.

If the user wishes to choose a different directory where the predictions will be saved he can specify it with the `-o` or `--output` flag.

By default the Phyloformer model used for inference is the one trained on simulations based on the PAM model of evolution, a different model to use can be specified with the `-m` or `--model` flag.
You can use a pretained model (either `seqgen` or [`evosimz`](https://gitlab.com/ztzou/phydl/-/tree/master/evosimz))  as well as your own trained phyloformer instances by specifying a path to a pytorch file containing the trained model. 

Finally, if an NVIDIA gpu is available, the `-g` or `--gpu` flag allows to exploit it offering a great speed up in inference. _(This flag will also work with M1 and M2 GPUs on newer Mac devices using the Metal API)_

## Simulations and training

To train the network one needs to simulate phylogenetic trees and alignments of sequences evolving along them.

### Simulating the trees

The trees can be generated with

```
simulate_trees \
    --nleaves <number of leaves in each tree> (default 20) \
    --ntrees <number of trees> \
    --type <tree topology> (default uniform) \
    --output <output directory> \
    --branchlength <branch lenght distribution> (default uniform)
```

The currently supported types of tree topologies are uniform (as in the paper, sampling uniformly from the tree topologies having nleaves), and birth-death (the tree is generated through a birth death process with a birth_rate of 1 and a death_rate of 0.5).

The currently supported types of branch length distribution are uniform (as in the paper, branch lenghts sampled uniformly between 0.002 and 1), and
exponential (branch lenghts sampled from an exponential distribution with a $\lambda$ parameter of 0.15)

Therefore to train the network just as in the paper one can create the tree dataset simply with

```
simulate_trees --ntrees 100000 -o <output directory>
```

### Simulating the alignments

Currently the supported sequence simulator is [Seq-Gen](http://tree.bio.ed.ac.uk/software/seqgen/)

The alignments can be generated with

```
simulate_alignments \
    --input <input directory with the .nwk tree files>  \
    --output <output directory> \
    --length <length of the simulated sequences> (default 200) \
    --seqgen <path to Seq-Gen-1.3.4/source/> \
    --model <model of evolution> (default PAM)
```

the possible models of evolution being those supported by Seq-Gen.

Again, to follow the paper one can just do

```
simulate_alignments \
    --input <input directory with the .nwk tree files>  \
    --output <output directory> \
    --seqgen <path to Seq-Gen-1.3.4/source/>
```

### Creating a tensor dataset
The trees and alignments then need to be converted to tensors:

```
make_tensors \
    --treedir <input directory with the .nwk tree files> \
    --alidir <input directory with the corresponding .fasta alignment files>  \
    --output <output directory>
```

### Training the model
Finally one can train their own phyloformer instance on the previously generated tensors. 

```
train_phyloformer \
    --input <input directory with the training tensors> \
    --output <output directory where the models will be saved>  \
    --config <json configuration file with hyperparameters>  \
    --load <path to model to train further> (optional)
```

A default configuration file is made available in [`config.json`](./config.json), i.e. the one used to train the model in the paper.

## Reproducibility of the results in the paper

The datasets simulated using Seq-Gen are available at:

- PAM model: [https://plmbox.math.cnrs.fr/f/f5a2ed2667a841cba6f0/](https://plmbox.math.cnrs.fr/f/f5a2ed2667a841cba6f0/).
- WAG model: [https://plmbox.math.cnrs.fr/f/834248a35ba64752a6a4/](https://plmbox.math.cnrs.fr/f/834248a35ba64752a6a4/).

The simulations under the Evosimz model are available at [https://datadryad.org/stash/dataset/doi%253A10.5061%252Fdryad.rbnzs7h91](https://datadryad.org/stash/dataset/doi%253A10.5061%252Fdryad.rbnzs7h91).

## Disclaimer:

Phyloformer is still at the stage of the proof of concept. In
particular, we observed lower performances:

- For trees with more than 40 leaves. As discussed in the manuscript
  this will likely be fixed after we re-train on trees with larger
  evolutionary distances.

- For topologies generated under a birth-death model. The topologies
  used in the experiments reported in the manuscript were generated
  under the populate function of the ete3 package. We are still
  investigating the possible reasons for this discrepancy.

We are making the trained models available for transparency and would
be very interested to hear about cases where these models
underperform.
