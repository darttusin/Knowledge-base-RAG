LinearLR 
====================================================

*class* torch.optim.lr_scheduler. LinearLR ( *optimizer*  , *start_factor = 0.3333333333333333*  , *end_factor = 1.0*  , *total_iters = 5*  , *last_epoch = -1* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L664) 
:   Decays the learning rate of each parameter group by linearly changing small multiplicative factor. 

The multiplication is done until the number of epoch reaches a pre-defined milestone: total_iters.
Notice that such decay can happen simultaneously with other changes to the learning rate
from outside this scheduler. When last_epoch=-1, sets initial lr as lr. 

Parameters
:   * **optimizer** ( [*Optimizer*](../optim.html#torch.optim.Optimizer "torch.optim.Optimizer")  ) – Wrapped optimizer.
* **start_factor** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – The number we multiply learning rate in the first epoch.
The multiplication factor changes towards end_factor in the following epochs.
Default: 1./3.
* **end_factor** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – The number we multiply learning rate at the end of linear changing
process. Default: 1.0.
* **total_iters** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – The number of iterations that multiplicative factor reaches to 1.
Default: 5.
* **last_epoch** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – The index of the last epoch. Default: -1.

Example 

```
>>> # Assuming optimizer uses lr = 0.05 for all groups
>>> # lr = 0.003687  if epoch == 0
>>> # lr = 0.004875  if epoch == 1
>>> # lr = 0.006062  if epoch == 2
>>> # lr = 0.00725   if epoch == 3
>>> # ...
>>> # lr = 0.05      if epoch >= 40
>>> scheduler = LinearLR(optimizer, start_factor=0.05, total_iters=40)
>>> for epoch in range(100):
>>>     train(...)
>>>     validate(...)
>>>     scheduler.step()

```

![../_images/LinearLR.png](../_images/LinearLR.png)

get_last_lr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L168) 
:   Return last computed learning rate by current scheduler. 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ]

get_lr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L723) 
:   Compute the learning rate. 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ]

load_state_dict ( *state_dict* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L159) 
:   Load the scheduler’s state. 

Parameters
: **state_dict** ( [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  ) – scheduler state. Should be an object returned
from a call to [`state_dict()`](#torch.optim.lr_scheduler.LinearLR.state_dict "torch.optim.lr_scheduler.LinearLR.state_dict")  .

state_dict ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L149) 
:   Return the state of the scheduler as a [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  . 

It contains an entry for every variable in self.__dict__ which
is not the optimizer. 

Return type
:   [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")  ]

step ( *epoch = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L176) 
:   Perform a step.

