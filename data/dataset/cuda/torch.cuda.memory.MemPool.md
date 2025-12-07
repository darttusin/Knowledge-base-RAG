MemPool 
==================================================

*class* torch.cuda.memory. MemPool ( *allocator = None*  , *use_on_oom = False*  , *symmetric = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/memory.py#L1153) 
:   MemPool represents a pool of memory in a caching allocator. Currently,
it’s just the ID of the pool object maintained in the CUDACachingAllocator. 

Parameters
:   * **allocator** ( *torch._C._cuda_CUDAAllocator* *,* *optional*  ) – a
torch._C._cuda_CUDAAllocator object that can be used to
define how memory gets allocated in the pool. If [`allocator`](#torch.cuda.memory.MemPool.allocator "torch.cuda.memory.MemPool.allocator")  is `None`  (default), memory allocation follows the default/
current configuration of the CUDACachingAllocator.
* **use_on_oom** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – a bool that indicates if this pool can be used
as a last resort if a memory allocation outside of the pool fails due
to Out Of Memory. This is False by default.
* **symmetric** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – a bool that indicates if this pool is symmetrical
across ranks. This is False by default.

*property* allocator *: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") [ _cuda_CUDAAllocator ]* 
:   Returns the allocator this MemPool routes allocations to.

*property* id *: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") [ [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") , [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") ]* 
:   Returns the ID of this pool as a tuple of two ints.

*property* is_symmetric *: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")* 
:   Returns whether this pool is used for NCCL’s symmetric memory.

snapshot ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/memory.py#L1197) 
:   Return a snapshot of the CUDA memory allocator pool state across all
devices. 

Interpreting the output of this function requires familiarity with the
memory allocator internals. 

Note 

See [Memory management](../notes/cuda.html#cuda-memory-management)  for more details about GPU memory
management.

use_count ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/memory.py#L1193) 
:   Returns the reference count of this pool. 

Return type
:   [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")

