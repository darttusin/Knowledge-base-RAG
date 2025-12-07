ChainedScheduler 
====================================================================

*class* torch.optim.lr_scheduler. ChainedScheduler ( *schedulers*  , *optimizer = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L1123) 
:   Chains a list of learning rate schedulers. 

Takes in a sequence of chainable learning rate schedulers and calls their
step() functions consecutively in just one call to step(). 

Parameters
:   * **schedulers** ( *sequence*  ) – sequence of chained schedulers.
* **optimizer** ( [*Optimizer*](../optim.html#torch.optim.Optimizer "torch.optim.Optimizer") *,* *optional*  ) – Wrapped optimizer. Default: None.

Example 

```
>>> # Assuming optimizer uses lr = 0.05 for all groups
>>> # lr = 0.05      if epoch == 0
>>> # lr = 0.0450    if epoch == 1
>>> # lr = 0.0405    if epoch == 2
>>> # ...
>>> # lr = 0.00675   if epoch == 19
>>> # lr = 0.06078   if epoch == 20
>>> # lr = 0.05470   if epoch == 21
>>> scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=20)
>>> scheduler2 = ExponentialLR(optimizer, gamma=0.9)
>>> scheduler = ChainedScheduler([scheduler1, scheduler2], optimizer=optimizer)
>>> for epoch in range(100):
>>>     train(...)
>>>     validate(...)
>>>     scheduler.step()

```

![../_images/ChainedScheduler.png](../_images/ChainedScheduler.png)

get_last_lr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L168) 
:   Return last computed learning rate by current scheduler. 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ]

get_lr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L172) 
:   Compute learning rate using chainable form of the scheduler. 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ]

load_state_dict ( *state_dict* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L1214) 
:   Load the scheduler’s state. 

Parameters
: **state_dict** ( [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  ) – scheduler state. Should be an object returned
from a call to [`state_dict()`](#torch.optim.lr_scheduler.ChainedScheduler.state_dict "torch.optim.lr_scheduler.ChainedScheduler.state_dict")  .

state_dict ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L1194) 
:   Return the state of the scheduler as a [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  . 

It contains an entry for every variable in self.__dict__ which
is not the optimizer.
The wrapped scheduler states will also be saved. 

Return type
:   [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")  ]

step ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L1186) 
:   Perform a step.

