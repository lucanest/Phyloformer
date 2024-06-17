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
  author={Nesterenko Luca, Luc Blassel, , Boussau Bastien, Jacob Laurent},
  title={Phyloformer: towards fast, accurate and versatile phylogenetic reconstruction with deep neural networks},
  doi=...,
  year={2024},
  journal={bioRxiv}
}
```

![](https://github.com/lucanest/Phyloformer/blob/main/figures/sketch.png?raw=true)



## Running Phyloformer


### Installing dependencies

The easiest way to install the software is by creating a virutal environment using conda/mamba and then installing this module locally:

```
UPDATE UPDATE UPDATE!!
# Install mamba if you want to use it instead of conda
conda install -n base -c conda-forge mamba

# Clone the phyloformer repo
git clone https://github.com/lucanest/Phyloformer.git && cd Phyloformer

# Create the virtual env and install the phyloformer package inside
conda create -n phylo python=3.9
conda activate phylo
pip install .
```

### Using pre-trained models
All the named phyloformer models in the manuscript are given in the [`models`](./models/) directory:
- `PF_Base` trained with an MAE loss on LG+GC data
- `PF` fine-tuned from PF_Base with an MRE loss on LG+GC data
- `PF_Indel` fine-tuned from PF_Base with an MAE loss on LG+GC+Indels data
- `PF_Cherry` fine-tuned from PF_Base with an MAE loss on CherryML data
- `PF_SelReg` fine-tuned from PF_Base with an MAE loss on SelReg data

Use the [`infer_alns.py`](./infer_alns.py) script to infer some distance matrices from alignments using a trained Phyloformer model

### Training your own models

#### Simulating data
Simulate trees with [`SimulateAlterRescale.py`](./SimulateAlterRescale.py), if you want to simulate alignment with alisim use [`alisim.py`](./alisim.py). 
If you want to use Cherry to simulate alignments use the [`TODO`](todo). 

#### Running the training
Use the [`train_distributed`](./train_distributed.py) script to train or fine-tune a PF model on some data (Need lightning, will work on a SLURM env)

### Re-producing figures
Use the [`make_plots`](./make_plots.py) script to reproduce all paper figures. Link to donwload results data first (quite large...) ***TODO***

