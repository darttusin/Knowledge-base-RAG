torch.is_tensor 
===================================================================

torch. is_tensor ( *obj*  , */* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/__init__.py#L1108) 
:   Returns True if *obj* is a PyTorch tensor. 

Note that this function is simply doing `isinstance(obj, Tensor)`  .
Using that `isinstance`  check is better for typechecking with mypy,
and more explicit - so it’s recommended to use that instead of `is_tensor`  . 

Parameters
: **obj** ( [*object*](https://docs.python.org/3/library/functions.html#object "(in Python v3.13)")  ) – Object to test

Return type
:   *TypeIs*  [ [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ]

Example: 

```
>>> x = torch.tensor([1, 2, 3])
>>> torch.is_tensor(x)
True

```

