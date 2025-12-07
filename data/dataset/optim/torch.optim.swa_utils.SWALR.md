SWALR 
==============================================

*class* torch.optim.swa_utils. SWALR ( *optimizer*  , *swa_lr*  , *anneal_epochs = 10*  , *anneal_strategy = 'cos'*  , *last_epoch = -1* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/swa_utils.py#L371) 
:   Anneals the learning rate in each parameter group to a fixed value. 

This learning rate scheduler is meant to be used with Stochastic Weight
Averaging (SWA) method (see *torch.optim.swa_utils.AveragedModel* ). 

Parameters
:   * **optimizer** ( [*torch.optim.Optimizer*](../optim.html#torch.optim.Optimizer "torch.optim.Optimizer")  ) – wrapped optimizer
* **swa_lrs** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *or* [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  ) – the learning rate value for all param groups
together or separately for each group.
* **annealing_epochs** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – number of epochs in the annealing phase
(default: 10)
* **annealing_strategy** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – “cos” or “linear”; specifies the annealing
strategy: “cos” for cosine annealing, “linear” for linear annealing
(default: “cos”)
* **last_epoch** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the index of the last epoch (default: -1)

The [`SWALR`](#torch.optim.swa_utils.SWALR "torch.optim.swa_utils.SWALR")  scheduler can be used together with other
schedulers to switch to a constant learning rate late in the training
as in the example below. 

Example 

```
>>> loader, optimizer, model = ...
>>> lr_lambda = lambda epoch: 0.9
>>> scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,
>>>        lr_lambda=lr_lambda)
>>> swa_scheduler = torch.optim.swa_utils.SWALR(optimizer,
>>>        anneal_strategy="linear", anneal_epochs=20, swa_lr=0.05)
>>> swa_start = 160
>>> for i in range(300):
>>>      for input, target in loader:
>>>          optimizer.zero_grad()
>>>          loss_fn(model(input), target).backward()
>>>          optimizer.step()
>>>      if i > swa_start:
>>>          swa_scheduler.step()
>>>      else:
>>>          scheduler.step()

```

get_last_lr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L168) 
:   Return last computed learning rate by current scheduler. 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ]

get_lr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/swa_utils.py#L456) 
:   Get learning rate.

load_state_dict ( *state_dict* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L159) 
:   Load the scheduler’s state. 

Parameters
: **state_dict** ( [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  ) – scheduler state. Should be an object returned
from a call to [`state_dict()`](#torch.optim.swa_utils.SWALR.state_dict "torch.optim.swa_utils.SWALR.state_dict")  .

state_dict ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L149) 
:   Return the state of the scheduler as a [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  . 

It contains an entry for every variable in self.__dict__ which
is not the optimizer. 

Return type
:   [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")  ]

step ( *epoch = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L176) 
:   Perform a step.

