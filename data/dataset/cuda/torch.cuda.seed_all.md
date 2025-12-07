torch.cuda.seed_all 
===========================================================================

torch.cuda. seed_all ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/random.py#L153) 
:   Set the seed for generating random numbers to a random number on all GPUs. 

It’s safe to call this function if CUDA is not available; in that
case, it is silently ignored.

