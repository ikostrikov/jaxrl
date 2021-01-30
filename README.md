# Jax (Flax) Soft Actor Critic

This is a Jax (Flax) implementation of [Soft Actor Critic with learnable temperature](https://arxiv.org/abs/1812.05905).

The goal of this repository is to provide a simple and clean implementation to build research on top of. **Please do not use this repository for baseline results and use the original implementation of SAC instead.**

# Installation

Install and activate an Anaconda environment
```bash
conda env create -f environment.yml 
conda activate jax-rl
```

If you want to run this code on GPU, please follow instructions from [the official repository](https://github.com/google/jax).

Please follow [the instructions](https://github.com/openai/mujoco-py/pull/583/files) to build mujoco-py with fast headless GPU rendering.

# Run

OpenAI Gym MuJoCo tasks

```bash
python train.py --env_name=HalfCheetah-v2 --save_dir=./tmp/
```

DeepMind Control suite (--env-name=dmc-domain-task)

```bash
python train.py --env_name=dmc-cheetah-run --save_dir=./tmp/
```

# Tensorboard

Launch tensorboard to see training and evaluation logs

```bash
tensorboard --logdir=./tmp/
```

# Results

![gym](./learning_curves/images/results.png)

# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue. If you are not familiar with pull requests, please read [this documentation](https://opensource.com/article/19/7/create-pull-request-github).
