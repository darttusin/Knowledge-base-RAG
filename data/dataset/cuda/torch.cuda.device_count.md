torch.cuda.device_count 
===================================================================================

torch.cuda. device_count ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/__init__.py#L1022) 
:   Return the number of GPUs available. 

Note 

This API will NOT posion fork if NVML discovery succeeds.
See [Poison fork in multiprocessing](../notes/multiprocessing.html#multiprocessing-poison-fork-note)  for more details.

Return type
:   [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")

