torch.xpu.seed_all 
=========================================================================

torch.xpu. seed_all ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/random.py#L134) 
:   Set the seed for generating random numbers to a random number on all GPUs. 

It’s safe to call this function if XPU is not available; in that case, it is silently ignored.

