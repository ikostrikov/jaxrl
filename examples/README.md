# Run

OpenAI Gym MuJoCo tasks

```bash
python train.py --env_name=HalfCheetah-v2 --save_dir=./tmp/
```

DeepMind Control suite (--env-name=domain-task)

```bash
python train.py --env_name=cheetah-run --save_dir=./tmp/
```

For continuous control from pixels

```bash
MUJOCO_GL=egl python train_pixels.py --env_name=cheetah-run --save_dir=./tmp/
```

For offline RL

```bash
python train_offline.py --env_name=halfcheetah-expert-v0  --dataset_name=d4rl --save_dir=./tmp/
```

For RL finetuning

```bash
python train_finetuning.py --env_name=HalfCheetah-v2 --dataset_name=awac --save_dir=./tmp/
```
