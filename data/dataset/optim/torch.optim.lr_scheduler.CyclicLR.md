CyclicLR 
====================================================

*class* torch.optim.lr_scheduler. CyclicLR ( *optimizer*  , *base_lr*  , *max_lr*  , *step_size_up = 2000*  , *step_size_down = None*  , *mode = 'triangular'*  , *gamma = 1.0*  , *scale_fn = None*  , *scale_mode = 'cycle'*  , *cycle_momentum = True*  , *base_momentum = 0.8*  , *max_momentum = 0.9*  , *last_epoch = -1* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L1429) 
:   Sets the learning rate of each parameter group according to cyclical learning rate policy (CLR). 

The policy cycles the learning rate between two boundaries with a constant frequency,
as detailed in the paper [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)  .
The distance between the two boundaries can be scaled on a per-iteration
or per-cycle basis. 

Cyclical learning rate policy changes the learning rate after every batch. *step* should be called after a batch has been used for training. 

This class has three built-in policies, as put forth in the paper: 

* “triangular”: A basic triangular cycle without amplitude scaling.
* “triangular2”: A basic triangular cycle that scales initial amplitude by half each cycle.
* “exp_range”: A cycle that scales initial amplitude by <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mtext>
               gamma
              </mtext>
<mtext>
               cycle iterations
              </mtext>
</msup>
</mrow>
<annotation encoding="application/x-tex">
             text{gamma}^{text{cycle iterations}}
            </annotation>
</semantics>
</math> -->gamma cycle iterations text{gamma}^{text{cycle iterations}}gamma cycle iterations  at each cycle iteration.

This implementation was adapted from the github repo: [bckenstler/CLR](https://github.com/bckenstler/CLR) 

Parameters
:   * **optimizer** ( [*Optimizer*](../optim.html#torch.optim.Optimizer "torch.optim.Optimizer")  ) – Wrapped optimizer.
* **base_lr** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *or* [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  ) – Initial learning rate which is the
lower boundary in the cycle for each parameter group.
* **max_lr** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *or* [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  ) – Upper learning rate boundaries in the cycle
for each parameter group. Functionally,
it defines the cycle amplitude (max_lr - base_lr).
The lr at any cycle is the sum of base_lr
and some scaling of the amplitude; therefore
max_lr may not actually be reached depending on
scaling function.
* **step_size_up** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Number of training iterations in the
increasing half of a cycle. Default: 2000
* **step_size_down** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Number of training iterations in the
decreasing half of a cycle. If step_size_down is None,
it is set to step_size_up. Default: None
* **mode** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – One of {triangular, triangular2, exp_range}.
Values correspond to policies detailed above.
If scale_fn is not None, this argument is ignored.
Default: ‘triangular’
* **gamma** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – Constant in ‘exp_range’ scaling function:
gamma**(cycle iterations)
Default: 1.0
* **scale_fn** ( *function*  ) – Custom scaling policy defined by a single
argument lambda function, where
0 <= scale_fn(x) <= 1 for all x >= 0.
If specified, then ‘mode’ is ignored.
Default: None
* **scale_mode** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – {‘cycle’, ‘iterations’}.
Defines whether scale_fn is evaluated on
cycle number or cycle iterations (training
iterations since start of cycle).
Default: ‘cycle’
* **cycle_momentum** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If `True`  , momentum is cycled inversely
to learning rate between ‘base_momentum’ and ‘max_momentum’.
Default: True
* **base_momentum** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *or* [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  ) – Lower momentum boundaries in the cycle
for each parameter group. Note that momentum is cycled inversely
to learning rate; at the peak of a cycle, momentum is
‘base_momentum’ and learning rate is ‘max_lr’.
Default: 0.8
* **max_momentum** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *or* [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  ) – Upper momentum boundaries in the cycle
for each parameter group. Functionally,
it defines the cycle amplitude (max_momentum - base_momentum).
The momentum at any cycle is the difference of max_momentum
and some scaling of the amplitude; therefore
base_momentum may not actually be reached depending on
scaling function. Note that momentum is cycled inversely
to learning rate; at the start of a cycle, momentum is ‘max_momentum’
and learning rate is ‘base_lr’
Default: 0.9
* **last_epoch** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – The index of the last batch. This parameter is used when
resuming a training job. Since *step()* should be invoked after each
batch instead of after each epoch, this number represents the total
number of *batches*  computed, not the total number of epochs computed.
When last_epoch=-1, the schedule is started from the beginning.
Default: -1

Example 

```
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> scheduler = torch.optim.lr_scheduler.CyclicLR(
...     optimizer,
...     base_lr=0.01,
...     max_lr=0.1,
...     step_size_up=10,
... )
>>> data_loader = torch.utils.data.DataLoader(...)
>>> for epoch in range(10):
>>>     for batch in data_loader:
>>>         train_batch(...)
>>>         scheduler.step()

```

![../_images/CyclicLR.png](../_images/CyclicLR.png)

get_last_lr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L168) 
:   Return last computed learning rate by current scheduler. 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ]

get_lr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L1639) 
:   Calculate the learning rate at batch index. 

This function treats *self.last_epoch* as the last batch index. 

If *self.cycle_momentum* is `True`  , this function has a side effect of
updating the optimizer’s momentum. 

Return type
:   [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")  [ [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ]

load_state_dict ( *state_dict* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L1702) 
:   Load the scheduler’s state.

scale_fn ( *x* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L1620) 
:   Get the scaling policy. 

Return type
:   [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")

step ( *epoch = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L176) 
:   Perform a step.

