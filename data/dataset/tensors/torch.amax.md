torch.amax 
========================================================

torch. amax ( *input*  , *dim*  , *keepdim = False*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns the maximum value of each slice of the `input`  tensor in the given
dimension(s) `dim`  . 

Note 

The difference between `max`  / `min`  and `amax`  / `amin`  is:
:   * `amax`  / `amin`  supports reducing on multiple dimensions,
* `amax`  / `amin`  does not return indices.

Both `max`  / `min`  and `amax`  / `amin`  evenly distribute gradients between equal values
when there are multiple input elements with the same minimum or maximum value.

If `keepdim`  is `True`  , the output tensor is of the same size
as `input`  except in the dimension(s) `dim`  where it is of size 1.
Otherwise, `dim`  is squeezed (see [`torch.squeeze()`](torch.squeeze.html#torch.squeeze "torch.squeeze")  ), resulting in the
output tensor having 1 (or `len(dim)`  ) fewer dimension(s). 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *of* *ints* *,* *optional*  ) – the dimension or dimensions to reduce.
If `None`  , all dimensions are reduced.
* **keepdim** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether the output tensor has `dim`  retained or not. Default: `False`  .

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 0.8177,  1.4878, -0.2491,  0.9130],
        [-0.7158,  1.1775,  2.0992,  0.4817],
        [-0.0053,  0.0164, -1.3738, -0.0507],
        [ 1.9700,  1.1106, -1.0318, -1.0816]])
>>> torch.amax(a, 1)
tensor([1.4878, 2.0992, 0.0164, 1.9700])

```

