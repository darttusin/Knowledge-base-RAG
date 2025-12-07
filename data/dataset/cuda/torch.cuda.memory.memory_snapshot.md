torch.cuda.memory.memory_snapshot 
=======================================================================================================

torch.cuda.memory. memory_snapshot ( *mempool_id = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/memory.py#L617) 
:   Return a snapshot of the CUDA memory allocator state across all devices. 

Interpreting the output of this function requires familiarity with the
memory allocator internals. 

Note 

See [Memory management](../notes/cuda.html#cuda-memory-management)  for more details about GPU memory
management.

