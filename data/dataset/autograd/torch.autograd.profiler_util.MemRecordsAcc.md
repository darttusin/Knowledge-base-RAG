MemRecordsAcc 
==============================================================

*class* torch.autograd.profiler_util. MemRecordsAcc ( *mem_records* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/autograd/profiler_util.py#L758) 
:   Acceleration structure for accessing mem_records in interval. 

in_interval ( *start_us*  , *end_us* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/autograd/profiler_util.py#L769) 
:   Return all records in the given interval
To maintain backward compatibility, convert us to ns in function

