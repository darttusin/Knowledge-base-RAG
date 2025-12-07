set_multithreading_enabled 
==========================================================================================

*class* torch.autograd.grad_mode. set_multithreading_enabled ( *mode* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/autograd/grad_mode.py#L299) 
:   Context-manager that sets multithreaded backwards on or off. 

`set_multithreading_enabled`  will enable or disable multithreaded backwards based on its argument `mode`  .
It can be used as a context-manager or as a function. 

This context manager is thread local; it will not affect computation
in other threads. 

Parameters
: **mode** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Flag whether to enable multithreaded backwards ( `True`  ), or disable
( `False`  ).

Note 

This API does not apply to [forward-mode AD](../autograd.html#forward-mode-ad)  .

clone ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/autograd/grad_mode.py#L328) 
:   Create a copy of this class 

Return type
:   [*set_multithreading_enabled*](#torch.autograd.grad_mode.set_multithreading_enabled "torch.autograd.grad_mode.set_multithreading_enabled")

