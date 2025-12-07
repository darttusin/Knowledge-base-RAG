torch.logical_xor 
=======================================================================

torch. logical_xor ( *input : [Tensor](../tensors.html#torch.Tensor "torch.Tensor")*  , *other : [Tensor](../tensors.html#torch.Tensor "torch.Tensor")*  , *** , *out : Optional [ [Tensor](../tensors.html#torch.Tensor "torch.Tensor") ]* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the element-wise logical XOR of the given input tensors. Zeros are treated as `False`  and nonzeros are
treated as `True`  . 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **other** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the tensor to compute XOR with

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.logical_xor(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
tensor([False, False,  True])
>>> a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
>>> b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
>>> torch.logical_xor(a, b)
tensor([ True,  True, False, False])
>>> torch.logical_xor(a.double(), b.double())
tensor([ True,  True, False, False])
>>> torch.logical_xor(a.double(), b)
tensor([ True,  True, False, False])
>>> torch.logical_xor(a, b, out=torch.empty(4, dtype=torch.bool))
tensor([ True,  True, False, False])

```

