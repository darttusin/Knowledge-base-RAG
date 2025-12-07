torch.nn.utils.remove_spectral_norm 
============================================================================================================

torch.nn.utils. remove_spectral_norm ( *module*  , *name = 'weight'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/utils/spectral_norm.py#L338) 
:   Remove the spectral normalization reparameterization from a module. 

Parameters
:   * **module** ( [*Module*](torch.nn.Module.html#torch.nn.Module "torch.nn.Module")  ) – containing module
* **name** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – name of weight parameter

Return type
:   *T_module*

Example 

```
>>> m = spectral_norm(nn.Linear(40, 10))
>>> remove_spectral_norm(m)

```

