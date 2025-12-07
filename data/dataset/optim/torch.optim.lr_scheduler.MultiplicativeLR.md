MultiplicativeLR 
====================================================================

*class* torch.optim.lr_scheduler. MultiplicativeLR ( *optimizer*  , *lr_lambda*  , *last_epoch = -1* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L375) 
:   Multiply the learning rate of each parameter group by the factor given in the specified function. 

When last_epoch=-1, set initial lr as lr. 

Parameters
:   * **optimizer** ( [*Optimizer*](../optim.html#torch.optim.Optimizer "torch.optim.Optimizer")  ) – Wrapped optimizer.
* **lr_lambda** ( *function* *or* [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  ) – A function which computes a multiplicative
factor given an integer parameter epoch, or a list of such
functions, one for each group in optimizer.param_groups.
* **last_epoch** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – The index of last epoch. Default: -1.

Example 

```
>>> lmbda = lambda epoch: 0.95
>>> scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
>>> for epoch in range(100):
>>>     train(...)
>>>     validate(...)
>>>     scheduler.step()

```

![../_images/MultiplicativeLR.png](../_images/MultiplicativeLR.png)

get_last_lr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L168) 
:   Return last computed learning rate by current scheduler. 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ]

get_lr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L463) 
:   Compute the learning rate of each parameter group. 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ]

load_state_dict ( *state_dict* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L445) 
:   Load the scheduler’s state. 

Parameters
: **state_dict** ( [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  ) – scheduler state. Should be an object returned
from a call to [`state_dict()`](#torch.optim.lr_scheduler.MultiplicativeLR.state_dict "torch.optim.lr_scheduler.MultiplicativeLR.state_dict")  .

state_dict ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L423) 
:   Return the state of the scheduler as a [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  . 

It contains an entry for every variable in self.__dict__ which
is not the optimizer.
The learning rate lambda functions will only be saved if they are callable objects
and not if they are functions or lambdas. 

Return type
:   [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")  ]

step ( *epoch = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L176) 
:   Perform a step.

