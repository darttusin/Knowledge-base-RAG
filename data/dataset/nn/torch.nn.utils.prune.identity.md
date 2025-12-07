torch.nn.utils.prune.identity 
==============================================================================================

torch.nn.utils.prune. identity ( *module*  , *name* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/utils/prune.py#L823) 
:   Apply pruning reparametrization without pruning any units. 

Applies pruning reparametrization to the tensor corresponding to the
parameter called `name`  in `module`  without actually pruning any
units. Modifies module in place (and also return the modified module)
by: 

1. adding a named buffer called `name+'_mask'`  corresponding to the
binary mask applied to the parameter `name`  by the pruning method.
2. replacing the parameter `name`  by its pruned version, while the
original (unpruned) parameter is stored in a new parameter named `name+'_orig'`  .

Note 

The mask is a tensor of ones.

Parameters
:   * **module** ( [*nn.Module*](torch.nn.Module.html#torch.nn.Module "torch.nn.Module")  ) – module containing the tensor to prune.
* **name** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – parameter name within `module`  on which pruning
will act.

Returns
:   modified (i.e. pruned) version of the input module

Return type
:   module ( [nn.Module](torch.nn.Module.html#torch.nn.Module "torch.nn.Module")  )

Examples 

```
>>> m = prune.identity(nn.Linear(2, 3), "bias")
>>> print(m.bias_mask)
tensor([1., 1., 1.])

```

