torch.permute 
==============================================================

torch. permute ( *input*  , *dims* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a view of the original tensor `input`  with its dimensions permuted. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **dims** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *of* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – The desired ordering of dimensions

Example 

```
>>> x = torch.randn(2, 3, 5)
>>> x.size()
torch.Size([2, 3, 5])
>>> torch.permute(x, (2, 0, 1)).size()
torch.Size([5, 2, 3])

```

