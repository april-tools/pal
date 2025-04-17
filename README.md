# PAL - A Probabilistic Neuro-symbolic Layer for Algebraic Constraint Satisfaction
[![Python application](https://github.com/april-tools/pal/actions/workflows/python-app.yml/badge.svg)](https://github.com/april-tools/pal/actions/workflows/python-app.yml)
This repo will contain the code for PAL, a probabilistic neuro-symbolic layer for algebraic constraint satisfaction.

# Installation

Just run:
```bash
./setup.sh
```
And you're ready to go!

# Constrained Stanford Drone Dataset

We provide an example script how to train a simple MLP on the constrained SDD-dataset. A model can be trained like this:

```bash
python pal/training/train_mlp_sdd.py --epochs 10 --init_last_layer_positive --seed 1744909132
```

This should result in a (mean) test log-likelihood of `-1.9800`.

# GASP!
The dependency was added via subtree from https://github.com/april-tools/gasp.git into pal/wmi/gasp!
update via:
```bash
git subtree pull --prefix pal/wmi/gasp https://github.com/april-tools/gasp.git main --squash
```
push via:
```bash
git subtree push --prefix pal/wmi/gasp https://github.com/april-tools/gasp.git main
```