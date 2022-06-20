

<p align="center">
  <img src="https://github.com/lucanest/Phyloformer/blob/main/phyloformer.png">
</p>

# Phyloformer: Fast and accurate Phylogeny estimation<br/> with self-attention Networks

- Luca Nesterenko
- Bastien Boussau
- Laurent Jacob

This repository contains the scripts, data, and plots for [the paper](https://arxiv.org/abs/???):


```bibtex
@article{Nesterenko2022phyloformer,
  author={Nesterenko Luca, Boussau Bastien, Jacob Laurent},
  title={PHYLOFORMER: FAST AND ACCURATE PHYLOGENY ESTIMATION
WITH SELF-ATTENTION NETWORKS},
  year={2022},
  doi={?},
  url={?},
  journal={bioRxiv}
}
```

![](sketch.png)

## Project structure

- 
- 
- 
- `LICENSE`: this repository is under the MIT license.

## Install
### First, install mamba:
```
conda install -n base -c conda-forge mamba
```

### To install it:
```
mamba env create -f environment.yml
```

### To run it:


`conda activate phylo`

### Test run

## Usage

## Tests and benchmarking
The data simulated using seq-gen is available at:

- PAM model: [https://plmbox.math.cnrs.fr/f/5bd0f367da8e4fad8d72/](https://plmbox.math.cnrs.fr/f/5bd0f367da8e4fad8d72/)
- WAG model: git@github.com:lucanest/Phyloformer.git

## Disclaimer:

Phyloformer is still at the stage of the proof of concept. In
particular, we observed lower performances:

- For trees with more than 40 leaves. As discussed in the manuscript
  this will likely be fixed after we re-train on trees with larger
  evolutionary distances.

- For topologies generated under a birth-death model. The topologies
  used in the experiments reported in the manuscript were generated
  under the populate function of the ete3 package. We are still
  investigating the possible cause of the reasons of this discrepancy.

We are making the trained models available for transparency and would
be very interested to hear about cases where these models
underperform.
