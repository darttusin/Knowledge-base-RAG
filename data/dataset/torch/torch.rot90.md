torch.rot90 
==========================================================

torch. rot90 ( *input*  , *k = 1*  , *dims = (0, 1)* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Rotate an n-D tensor by 90 degrees in the plane specified by dims axis.
Rotation direction is from the first towards the second axis if k > 0, and from the second towards the first for k < 0. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **k** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – number of times to rotate. Default value is 1
* **dims** ( *a list* *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  ) – axis to rotate. Default value is [0, 1]

Example: 

```
>>> x = torch.arange(4).view(2, 2)
>>> x
tensor([[0, 1],
        [2, 3]])
>>> torch.rot90(x, 1, [0, 1])
tensor([[1, 3],
        [0, 2]])

>>> x = torch.arange(8).view(2, 2, 2)
>>> x
tensor([[[0, 1],
         [2, 3]],

        [[4, 5],
         [6, 7]]])
>>> torch.rot90(x, 1, [1, 2])
tensor([[[1, 3],
         [0, 2]],

        [[5, 7],
         [4, 6]]])

```

