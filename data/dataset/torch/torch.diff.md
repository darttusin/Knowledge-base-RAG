torch.diff 
========================================================

torch. diff ( *input*  , *n = 1*  , *dim = -1*  , *prepend = None*  , *append = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the n-th forward difference along the given dimension. 

The first-order differences are given by *out[i] = input[i + 1] - input[i]* . Higher-order
differences are calculated by using [`torch.diff()`](#torch.diff "torch.diff")  recursively. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the tensor to compute the differences on
* **n** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – the number of times to recursively compute the difference
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – the dimension to compute the difference along.
Default is the last dimension.
* **prepend** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – values to prepend or append to `input`  along `dim`  before computing the difference.
Their dimensions must be equivalent to that of input, and their shapes
must match input’s shape except on `dim`  .
* **append** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – values to prepend or append to `input`  along `dim`  before computing the difference.
Their dimensions must be equivalent to that of input, and their shapes
must match input’s shape except on `dim`  .

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.tensor([1, 3, 2])
>>> torch.diff(a)
tensor([ 2, -1])
>>> b = torch.tensor([4, 5])
>>> torch.diff(a, append=b)
tensor([ 2, -1,  2,  1])
>>> c = torch.tensor([[1, 2, 3], [3, 4, 5]])
>>> torch.diff(c, dim=0)
tensor([[2, 2, 2]])
>>> torch.diff(c, dim=1)
tensor([[1, 1],
        [1, 1]])

```

