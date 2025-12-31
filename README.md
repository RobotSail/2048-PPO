# REINFORCE 2048

This repo implements an agent that learns to play 2048 using policy gradient (REINFORCE).

You can train a model to play the 2048 game with the `train.py` script, and you can use the `viz_server.py` script to watch the agent play the game as it learns.


This command should give you decent resuts, although REINFORCE is a very high-variance method so
it becomes sensitive to your choice of hyperparameters:

```bash
 python train.py train --batch-size=4  --steps=20000 --lr 0.001 --critic-lr 1e-4  -h 196  --gamma 0.99 --entropy 0.02 --smoothness 0.0 --tile-bonus 0.0 --print-freq 5  --corner 0.0     --points 0.10 --show-last-steps 0         --viz-dir viz_data         --mono  1.0 --model-type mlp    --critic 0.2       --rtg-beta 0.99    --wandb      --eval-freq 100 --emptiness 0.0        --warmup-steps 10 --upsample-ratio 0.25
```

## Model Architectures

This repo implements two model architectures:

### MLP

The MLP architecture is a fairly simple MLP with a backbone consisting of two hidden layers followed by an action head and a value head.

The value head acts as a critic to estimate the advantage function.


### URM

The [URM](https://arxiv.org/abs/2512.14693) architecture is a universal reasoning model that is a variant of the Transformer architecture.

It has a higher parameter count but in theory should be able to learn more complex patterns.



## Results

The highest score I've been able to achieve with the MLP architecture so far has been around 16-18k, though I haven't ran it for long enough to get a 2048 tile.
