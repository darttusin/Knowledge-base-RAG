SequentialLR 
============================================================

*class* torch.optim.lr_scheduler. SequentialLR ( *optimizer*  , *schedulers*  , *milestones*  , *last_epoch = -1* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L807) 
:   Contains a list of schedulers expected to be called sequentially during the optimization process. 

Specifically, the schedulers will be called according to the milestone points, which should provide exact
intervals by which each scheduler should be called at a given epoch. 

Parameters
:   * **optimizer** ( [*Optimizer*](../optim.html#torch.optim.Optimizer "torch.optim.Optimizer")  ) – Wrapped optimizer.
* **schedulers** ( [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  ) – List of chained schedulers.
* **milestones** ( [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  ) – List of integers that reflects milestone points.
* **last_epoch** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – The index of last epoch. Default: -1.

Example 

```
>>> # Assuming optimizer uses lr = 0.05 for all groups
>>> # lr = 0.005     if epoch == 0
>>> # lr = 0.005     if epoch == 1
>>> # lr = 0.005     if epoch == 2
>>> # ...
>>> # lr = 0.05      if epoch == 20
>>> # lr = 0.045     if epoch == 21
>>> # lr = 0.0405    if epoch == 22
>>> scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=20)
>>> scheduler2 = ExponentialLR(optimizer, gamma=0.9)
>>> scheduler = SequentialLR(
...     optimizer,
...     schedulers=[scheduler1, scheduler2],
...     milestones=[20],
... )
>>> for epoch in range(100):
>>>     train(...)
>>>     validate(...)
>>>     scheduler.step()

```

![../_images/SequentialLR.png](../_images/SequentialLR.png)

get_last_lr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L168) 
:   Return last computed learning rate by current scheduler. 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ]

get_lr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L172) 
:   Compute learning rate using chainable form of the scheduler. 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ]

load_state_dict ( *state_dict* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L942) 
:   Load the scheduler’s state. 

Parameters
: **state_dict** ( [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  ) – scheduler state. Should be an object returned
from a call to [`state_dict()`](#torch.optim.lr_scheduler.SequentialLR.state_dict "torch.optim.lr_scheduler.SequentialLR.state_dict")  .

recursive_undo ( *sched = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L897) 
:   Recursively undo any step performed by the initialisation of
schedulers.

state_dict ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L922) 
:   Return the state of the scheduler as a [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  . 

It contains an entry for every variable in self.__dict__ which
is not the optimizer.
The wrapped scheduler states will also be saved. 

Return type
:   [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")  ]

step ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L910) 
:   Perform a step.

