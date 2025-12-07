torch.reshape 
==============================================================

torch. reshape ( *input*  , *shape* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a tensor with the same data and number of elements as `input`  ,
but with the specified shape. When possible, the returned tensor will be a view
of `input`  . Otherwise, it will be a copy. Contiguous inputs and inputs
with compatible strides can be reshaped without copying, but you should not
depend on the copying vs. viewing behavior. 

See [`torch.Tensor.view()`](torch.Tensor.view.html#torch.Tensor.view "torch.Tensor.view")  on when it is possible to return a view. 

A single dimension may be -1, in which case it’s inferred from the remaining
dimensions and the number of elements in `input`  . 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the tensor to be reshaped
* **shape** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *of* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the new shape

Example: 

```
>>> a = torch.arange(4.)
>>> torch.reshape(a, (2, 2))
tensor([[ 0.,  1.],
        [ 2.,  3.]])
>>> b = torch.tensor([[0, 1], [2, 3]])
>>> torch.reshape(b, (-1,))
tensor([ 0,  1,  2,  3])

```

