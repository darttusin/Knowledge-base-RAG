torch.max 
======================================================

torch. max ( *input*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns the maximum value of all elements in the `input`  tensor. 

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.randn(1, 3)
>>> a
tensor([[ 0.6763,  0.7445, -2.2369]])
>>> torch.max(a)
tensor(0.7445)

```

torch. max ( *input*  , *dim*  , *keepdim = False*  , *** , *out = None* )
:

Returns a namedtuple `(values, indices)`  where `values`  is the maximum
value of each row of the `input`  tensor in the given dimension `dim`  . And `indices`  is the index location of each maximum value found
(argmax). 

If `keepdim`  is `True`  , the output tensors are of the same size
as `input`  except in the dimension `dim`  where they are of size 1.
Otherwise, `dim`  is squeezed (see [`torch.squeeze()`](torch.squeeze.html#torch.squeeze "torch.squeeze")  ), resulting
in the output tensors having 1 fewer dimension than `input`  . 

Note 

If there are multiple maximal values in a reduced row then
the indices of the first maximal value are returned.

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *of* *ints* *,* *optional*  ) – the dimension or dimensions to reduce.
If `None`  , all dimensions are reduced.
* **keepdim** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether the output tensor has `dim`  retained or not. Default: `False`  .

Keyword Arguments
: **out** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – the result tuple of two output tensors (max, max_indices)

Example: 

```
>>> a = torch.randn(4, 4)
>>> a
tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
        [ 1.1949, -1.1127, -2.2379, -0.6702],
        [ 1.5717, -0.9207,  0.1297, -1.8768],
        [-0.6172,  1.0036, -0.6060, -0.2432]])
>>> torch.max(a, 1)
torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1]))
>>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
>>> a.max(dim=1, keepdim=True)
torch.return_types.max(
values=tensor([[2.], [4.]]),
indices=tensor([[1], [1]]))
>>> a.max(dim=1, keepdim=False)
torch.return_types.max(
values=tensor([2., 4.]),
indices=tensor([1, 1]))

```

torch. max ( *input*  , *other*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor")
:

See [`torch.maximum()`](torch.maximum.html#torch.maximum "torch.maximum")  .

