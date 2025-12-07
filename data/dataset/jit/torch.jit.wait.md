torch.jit.wait 
================================================================

torch.jit. wait ( *future* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/jit/_async.py#L102) 
:   Force completion of a *torch.jit.Future[T]* asynchronous task, returning the result of the task. 

See [`fork()`](torch.jit.fork.html#torch.jit.fork "torch.jit.fork")  for docs and examples.
:param future: an asynchronous task reference, created through *torch.jit.fork* :type future: torch.jit.Future[T] 

Returns
:   the return value of the completed task

Return type
:   *T*

