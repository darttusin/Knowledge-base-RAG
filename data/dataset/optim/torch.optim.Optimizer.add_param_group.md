torch.optim.Optimizer.add_param_group 
================================================================================================================

Optimizer. add_param_group ( *param_group* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py#L1066) 
:   Add a param group to the [`Optimizer`](../optim.html#torch.optim.Optimizer "torch.optim.Optimizer")  s *param_groups* . 

This can be useful when fine tuning a pre-trained network as frozen layers can be made
trainable and added to the [`Optimizer`](../optim.html#torch.optim.Optimizer "torch.optim.Optimizer")  as training progresses. 

Parameters
: **param_group** ( [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  ) – Specifies what Tensors should be optimized along with group
specific optimization options.

