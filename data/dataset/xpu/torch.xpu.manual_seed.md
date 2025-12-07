torch.xpu.manual_seed 
===============================================================================

torch.xpu. manual_seed ( *seed* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/random.py#L76) 
:   Set the seed for generating random numbers for the current GPU. 

It’s safe to call this function if XPU is not available; in that case, it is silently ignored. 

Parameters
: **seed** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – The desired seed.

Warning 

If you are working with a multi-GPU model, this function is insufficient
to get determinism. To seed all GPUs, use [`manual_seed_all()`](torch.xpu.manual_seed_all.html#torch.xpu.manual_seed_all "torch.xpu.manual_seed_all")  .

