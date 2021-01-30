# Jax (Flax) Soft Actor Critic

This is a Jax (Flax) implementation of [Soft Actor Critic with learnable temperature](https://arxiv.org/abs/1812.05905).

The goal of this repository is to provide a simple and clean implementation to build research on top of. **Please do not use this repository for baseline results and use the original implementation of SAC instead.**

# Installation

Install and activate an Anaconda environment
```bash
conda env create -f environment.yml 
conda activate flax-sac
```

If you want to run this code on GPU, please follow instructions from [the official repository](https://github.com/google/jax).

# Run

```bash
python main.py --env_name=HalfCheetah-v2 --save_dir=./tmp/
```

# Results

![gym](./images/results.png)
