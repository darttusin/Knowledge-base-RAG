torch.cuda.set_sync_debug_mode 
===================================================================================================

torch.cuda. set_sync_debug_mode ( *debug_mode* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/__init__.py#L1169) 
:   Set the debug mode for cuda synchronizing operations. 

Parameters
: **debug_mode** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – if “default” or 0, don’t error or warn on synchronizing operations,
if “warn” or 1, warn on synchronizing operations, if “error” or 2, error out synchronizing operations.

Warning 

This is an experimental feature, and not all synchronizing operations will trigger warning or error. In
particular, operations in torch.distributed and torch.sparse namespaces are not covered yet.

