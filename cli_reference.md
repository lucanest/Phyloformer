# CLI reference

## Inferring trees

```shell
usage: predict [-h] [-o OUTPUT] [-m MODEL] [-g] [-d] alidir

Predict phylogenetic trees from MSAs using the Phyloformer neural network

positional arguments:
  alidir                path to input directory containing the .fasta
                        alignments

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        path to the output directory were the .tree tree files
                        will be saved (default: alidir)
  -m MODEL, --model MODEL
                        path to the NN model's state dictionary. Possible
                        values are: [seqgen, evosimz, <path/to/model.pt>]
                        (default: seqgen)
  -g, --gpu             use the GPU for inference (default: false)
  -d, --dm              save predicted distance matrix (default: false)
```

## Training a model

```shell
usage: train_phyloformer [-h] -i INPUT [-v VALIDATION] -c CONFIG [-o OUTPUT]
                         [-l CHECKPOINT] [-g {tensorboard,file,both}]
                         [--logfile LOGFILE]

Train a phyloformer model

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        /path/ to input directory containing the the tensor
                        pairs on which the model will be trained
  -v VALIDATION, --validation VALIDATION
                        /path/ to input directory containing the the tensor
                        pairs on which the model will be evaluated. If left
                        empty 10% of the training set will be used as
                        validation data.
  -c CONFIG, --config CONFIG
                        /path/ to the configuration json file for the
                        hyperparameters
  -o OUTPUT, --output OUTPUT
                        /path/ to output directory where the model parameters
                        and the metrics will be saved (default: current
                        directory)
  -l CHECKPOINT, --load CHECKPOINT
                        Load training checkpoint
  -g {tensorboard,file,both}, --log {tensorboard,file,both}
                        How to log training process
  --logfile LOGFILE     path to save log at

```

## Simulating trees

```shell
usage: simulate_trees [-h] [-n NTREES] [-l NLEAVES] [-t TOPO] [-o OUTPUT]
                      [-b BL]

optional arguments:
  -h, --help            show this help message and exit
  -n NTREES, --ntrees NTREES
                        number of trees (default: 20)
  -l NLEAVES, --nleaves NLEAVES
                        number of leaves (default: 20)
  -t TOPO, --topology TOPO
                        tree topology. Choices: ['birth-death', 'uniform']
                        (default: uniform)
  -o OUTPUT, --output OUTPUT
                        path to the output directory were the .nwk tree files
                        will be saved (default: .)
  -b BL, --branchlength BL
                        branch length distribution. Choices: ['exponential',
                        'uniform'] (default: uniform)
```

## Simulating alignments

```shell
usage: simulate_alignments [-h] -i INPUT -o OUTPUT -s SEQGEN [-l LENGTH]
                           [-m MODEL]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to input directory containing the .nwk tree files
  -o OUTPUT, --output OUTPUT
                        path to output directory
  -s SEQGEN, --seqgen SEQGEN
                        path to the seq-gen executable
  -l LENGTH, --length LENGTH
                        length of the sequences in the alignments (default:
                        200)
  -m MODEL, --model MODEL
                        model of evolution. Allowed values: [JTT, WAG, PAM,
                        BLOSUM, MTREV, CPREV45, MTART, LG, HIVB, GENERAL]
                        (default: PAM)
```

## Compute tensors

```shell
usage: make_tensors [-h] -t TREEDIR -a ALIDIR [-o OUTPUT]

Generate a tensor training set from trees and MSAs

optional arguments:
  -h, --help            show this help message and exit
  -t TREEDIR, --treedir TREEDIR
                        path to input directory containing the .nwk tree files
  -a ALIDIR, --alidir ALIDIR
                        path to input directory containing corresponding
                        .fasta alignments
  -o OUTPUT, --output OUTPUT
                        path to output directory (default: current directory)
```

## Evaluate predictions

```shell
usage: evaluate [-h] -t TRUE -p PREDICTIONS

Compute the RF distance between predicted trees and true trees.

optional arguments:
  -h, --help            show this help message and exit
  -t TRUE, --true TRUE  path to directory containing true trees in .nwk format
  -p PREDICTIONS, --predictions PREDICTIONS
                        path to directory containing predicted trees in .nwk
                        format
```
