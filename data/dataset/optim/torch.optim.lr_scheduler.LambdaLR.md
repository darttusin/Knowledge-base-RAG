LambdaLR 
====================================================

*class* torch.optim.lr_scheduler. LambdaLR ( *optimizer*  , *lr_lambda*  , *last_epoch = -1* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L266) 
:   Sets the initial learning rate. 

The learning rate of each parameter group is set to the initial lr
times a given function. When last_epoch=-1, sets initial lr as lr. 

Parameters
:   * **optimizer** ( [*Optimizer*](../optim.html#torch.optim.Optimizer "torch.optim.Optimizer")  ) – Wrapped optimizer.
* **lr_lambda** ( *function* *or* [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  ) – A function which computes a multiplicative
factor given an integer parameter epoch, or a list of such
functions, one for each group in optimizer.param_groups.
* **last_epoch** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – The index of last epoch. Default: -1.

Example 

```
>>> # Assuming optimizer has two groups.
>>> num_epochs = 100
>>> lambda1 = lambda epoch: epoch // 30
>>> lambda2 = lambda epoch: 0.95**epoch
>>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
>>> for epoch in range(num_epochs):
>>>     train(...)
>>>     validate(...)
>>>     scheduler.step()
>>>
>>> # Alternatively, you can use a single lambda function for all groups.
>>> scheduler = LambdaLR(opt, lr_lambda=lambda epoch: epoch // 30)
>>> for epoch in range(num_epochs):
>>>     train(...)
>>>     validate(...)
>>>     scheduler.step()

```

![../_images/LambdaLR.png](../_images/LambdaLR.png)

get_last_lr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L168) 
:   Return last computed learning rate by current scheduler. 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ]

get_lr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L364) 
:   Compute learning rate. 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ]

load_state_dict ( *state_dict* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L344) 
:   Load the scheduler’s state. 

When saving or loading the scheduler, please make sure to also save or load the state of the optimizer. 

Parameters
: **state_dict** ( [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  ) – scheduler state. Should be an object returned
from a call to [`state_dict()`](#torch.optim.lr_scheduler.LambdaLR.state_dict "torch.optim.lr_scheduler.LambdaLR.state_dict")  .

state_dict ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L320) 
:   Return the state of the scheduler as a [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  . 

It contains an entry for every variable in self.__dict__ which
is not the optimizer.
The learning rate lambda functions will only be saved if they are callable objects
and not if they are functions or lambdas. 

When saving or loading the scheduler, please make sure to also save or load the state of the optimizer. 

Return type
:   [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")  ]

step ( *epoch = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L176) 
:   Perform a step.

