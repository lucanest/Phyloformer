# CLI reference

## Inferring trees

```
> python infer_alns.py --help

usage: Infer evolutionnary distances with PhyloFormer [-h] [--output-dir OUTPUT_DIR] [--trees] checkpoint alignments

positional arguments:
  checkpoint            Path to model checkpoint to use
  alignments            Path to alignment to infer tree for

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Path to output distance matrix
  --trees, -t           Output NJ trees
```

## Training a model

```
> python train_distributed.py --help

usage: train PF instance [-h] --train-trees TRAIN_TREES --train-alignments
                         TRAIN_ALIGNMENTS [--val-trees VAL_TREES]
                         [--val-alignments VAL_ALIGNMENTS]
                         [--train-regex TRAIN_REGEX] [--val-regex VAL_REGEX]
                         [--load-checkpoint LOAD_CHECKPOINT]
                         [--base-model BASE_MODEL] [--dropout DROPOUT]
                         [--nb-blocks NB_BLOCKS] [--embed-dim EMBED_DIM]
                         [--nb-heads NB_HEADS] [--nb-epochs NB_EPOCHS]
                         [--warmup-steps WARMUP_STEPS]
                         [--learning-rate LEARNING_RATE]
                         [--check-val-every CHECK_VAL_EVERY]
                         [--batch-size BATCH_SIZE] [--max-steps MAX_STEPS]
                         [--no-improvement-stop NO_IMPROVEMENT_STOP]
                         [--hard-loss-ceiling HARD_LOSS_CEILING]
                         [--output-dir OUTPUT_DIR] [--log-every LOG_EVERY]
                         [--project-name PROJECT_NAME] [--run-name RUN_NAME]
                         [--find-batch-size] [--profile]

optional arguments:
  -h, --help            show this help message and exit

data:
  Data IO parameters

  --train-trees TRAIN_TREES, -t TRAIN_TREES
                        Directory with training trees (default: None)
  --train-alignments TRAIN_ALIGNMENTS, -a TRAIN_ALIGNMENTS
                        Directory with training alignments (default: None)
  --val-trees VAL_TREES, -T VAL_TREES
                        Directory with validation trees (default: None)
  --val-alignments VAL_ALIGNMENTS, -A VAL_ALIGNMENTS
                        Directory with validation alignments (default: None)
  --train-regex TRAIN_REGEX, -r TRAIN_REGEX
                        Regex to filter training examples (default: None)
  --val-regex VAL_REGEX, -R VAL_REGEX
                        Regex to filter validation examples (default: None)

STARTING POINT:
  Model starting point, specifying one of these options will override
  options from the [ARCHITECTURE] parameter group.

  --load-checkpoint LOAD_CHECKPOINT, -c LOAD_CHECKPOINT
                        Path to checkpoint to resume training from (default:
                        None)
  --base-model BASE_MODEL, -m BASE_MODEL
                        Base model to fine-tune (default: None)

MODEL ARCHITECTURE:
  --dropout DROPOUT, -D DROPOUT
                        Dropout proportion (default: 0.0)
  --nb-blocks NB_BLOCKS, -b NB_BLOCKS
                        Number of PF blocks (default: 6)
  --embed-dim EMBED_DIM, -d EMBED_DIM
                        Number of embedding dimensions (default: 64)
  --nb-heads NB_HEADS, -H NB_HEADS
                        Number of attention heads (default: 4)

TRAINING:
  Training control parameters

  --nb-epochs NB_EPOCHS, -e NB_EPOCHS
                        Number of epochs to train for (default: 100)
  --warmup-steps WARMUP_STEPS, -w WARMUP_STEPS
                        Number of warmup steps (default: 5000)
  --learning-rate LEARNING_RATE, -l LEARNING_RATE
                        Traget starting learning rate (default: 0.0001)
  --check-val-every CHECK_VAL_EVERY, -C CHECK_VAL_EVERY
                        Check validation dataset every n steps (default:
                        10000)
  --batch-size BATCH_SIZE, -s BATCH_SIZE
                        Training batch size (default: 4)
  --max-steps MAX_STEPS, -M MAX_STEPS
                        Max number of training steps (default: None)
  --no-improvement-stop NO_IMPROVEMENT_STOP, -n NO_IMPROVEMENT_STOP
                        Number of checks with no improvement before stopping
                        early (default: 5)
  --hard-loss-ceiling HARD_LOSS_CEILING, -L HARD_LOSS_CEILING
                        Max value of loss over which the training stops
                        (default: 3.0)

LOGGING:
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Output directory to save losses and checkpoints
                        (default: .)
  --log-every LOG_EVERY, -E LOG_EVERY
                        Log training loss every n steps (default: 100)
  --project-name PROJECT_NAME, -p PROJECT_NAME
                        Project in which to save this run on WandB (default:
                        PHYLOFORMER_EXPERIMENTS)
  --run-name RUN_NAME, -N RUN_NAME
                        Name to give to the run on WandB (default: None)

UTILS:
  Utilities that are run instead of training

  --find-batch-size     Run the lightning batch_size finder (skips training)
                        (default: False)
  --profile             Run profiler for a few steps and exit (default: False)
```

## Simulating trees

```
> python simulate_trees.py --help

usage: simulate_trees.py [-h] [-n NTREES] [-t NTIPS] [--type {birth-death,uniform}] [-o OUTPUT] [--verbose VERBOSE]

optional arguments:
  -h, --help            show this help message and exit
  -n NTREES, --ntrees NTREES
                        Number of trees to simulate
  -t NTIPS, --ntips NTIPS
                        Size of the trees to simulate
  --type {birth-death,uniform}
                        Simulation methods for the trees: birth-death or uniform
  -o OUTPUT, --output OUTPUT
                        path to the output directory were the .nwk tree files will be saved
  --verbose VERBOSE
```

## Simulating alignments

```
> python alisim.py --help

usage: Alignment simulator [-h] [--outdir OUTDIR] [--length LENGTH] [--gamma GAMMA] [--substitution SUBSTITUTION]
                           [--custom-model CUSTOM_MODEL] [--no-summary] [--allow-duplicate-sequences] [--keep-logfiles]
                           [--max-attempts MAX_ATTEMPTS] [--processes PROCESSES]
                           trees

positional arguments:
  trees                 Path to the directory containing mewick trees

optional arguments:
  -h, --help            show this help message and exit
  --outdir OUTDIR, -o OUTDIR
                        Path to the output directory
  --length LENGTH, -l LENGTH
                        Length of the alignment
  --gamma GAMMA, -g GAMMA
                        Gamma model for between-site rate heterogeneity (G[n] for discrete gamma with n categories, GC for
                        continuous gamma)
  --substitution SUBSTITUTION, -s SUBSTITUTION
                        Protein substitution model: classical (LG, WAG, Dayhoff, Blosum62) or mixture (C10, ..., C60)
  --custom-model CUSTOM_MODEL, -c CUSTOM_MODEL
                        Path to a custom model definition in the nexus format (e.g. the UDM models in
                        github.com/dschrempf/EDCluster/Distributions/hogenom/*_lclr_iqtree.nex)
  --no-summary, -n      If specified suppress the output summarizing which simulation attempts have failed
  --allow-duplicate-sequences, -d
                        Allow duplicate sequences in the alignments
  --keep-logfiles, -k   Keep IQTree generated log files
  --max-attempts MAX_ATTEMPTS, -m MAX_ATTEMPTS
                        Maximum number of attempts to simulate alignment in case of duplicates
  --processes PROCESSES, -p PROCESSES
                        Number of threads for alisim to use.
```
