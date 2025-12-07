torch.optim.Optimizer.step 
========================================================================================

Optimizer. step ( *closure : [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") = None* ) → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py#L1057) 
Optimizer. step ( *closure : Callable [ [ ] , [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") ]* ) → [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")
:   Perform a single optimization step to update parameter. 

Parameters
: **closure** ( *Callable*  ) – A closure that reevaluates the model and
returns the loss. Optional for most optimizers.

