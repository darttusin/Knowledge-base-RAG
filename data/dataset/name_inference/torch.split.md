torch.split 
==========================================================

torch. split ( *tensor*  , *split_size_or_sections*  , *dim = 0* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/functional.py#L170) 
:   Splits the tensor into chunks. Each chunk is a view of the original tensor. 

If `split_size_or_sections`  is an integer type, then [`tensor`](torch.tensor.html#torch.tensor "torch.tensor")  will
be split into equally sized chunks (if possible). Last chunk will be smaller if
the tensor size along the given dimension `dim`  is not divisible by `split_size`  . 

If `split_size_or_sections`  is a list, then [`tensor`](torch.tensor.html#torch.tensor "torch.tensor")  will be split
into `len(split_size_or_sections)`  chunks with sizes in `dim`  according
to `split_size_or_sections`  . 

Parameters
:   * **tensor** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor to split.
* **split_size_or_sections** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *) or* *(* [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)") *(* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *)*  ) – size of a single chunk or
list of sizes for each chunk
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – dimension along which to split the tensor.

Return type
:   [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [ [torch.Tensor](../tensors.html#torch.Tensor "torch.Tensor")  , …]

Example: 

```
>>> a = torch.arange(10).reshape(5, 2)
>>> a
tensor([[0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]])
>>> torch.split(a, 2)
(tensor([[0, 1],
         [2, 3]]),
 tensor([[4, 5],
         [6, 7]]),
 tensor([[8, 9]]))
>>> torch.split(a, [1, 4])
(tensor([[0, 1]]),
 tensor([[2, 3],
         [4, 5],
         [6, 7],
         [8, 9]]))

```

