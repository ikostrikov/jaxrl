export WANDB_ENTITY=openrlbenchmark

xvfb-run -a python benchmark.py \
    --env_names HalfCheetah-v2 Hopper-v2 Humanoid-v2 InvertedDoublePendulum-v2 InvertedPendulum-v2 Pusher-v2 Reacher-v2 Swimmer-v2 Walker2d-v2 \
    --command "poetry run python train.py --track --save_video" \
    --num-seeds 10 \
    --workers 1
