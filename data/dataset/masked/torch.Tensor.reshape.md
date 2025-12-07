torch.Tensor.reshape 
============================================================================

Tensor. reshape ( ** shape* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a tensor with the same data and number of elements as `self`  but with the specified shape. This method returns a view if [`shape`](torch.Tensor.shape.html#torch.Tensor.shape "torch.Tensor.shape")  is
compatible with the current shape. See [`torch.Tensor.view()`](torch.Tensor.view.html#torch.Tensor.view "torch.Tensor.view")  on when it is
possible to return a view. 

See [`torch.reshape()`](torch.reshape.html#torch.reshape "torch.reshape") 

Parameters
: **shape** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *of* *ints* *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *...*  ) – the desired shape

