torch.xpu.manual_seed_all 
========================================================================================

torch.xpu. manual_seed_all ( *seed* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/random.py#L98) 
:   Set the seed for generating random numbers on all GPUs. 

It’s safe to call this function if XPU is not available; in that case, it is silently ignored. 

Parameters
: **seed** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – The desired seed.

