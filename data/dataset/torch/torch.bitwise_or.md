torch.bitwise_or 
=====================================================================

torch. bitwise_or ( *input : [Tensor](../tensors.html#torch.Tensor "torch.Tensor")*  , *other : [Tensor](../tensors.html#torch.Tensor "torch.Tensor")*  , *** , *out : Optional [ [Tensor](../tensors.html#torch.Tensor "torch.Tensor") ]* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the bitwise OR of `input`  and `other`  . The input tensor must be of
integral or Boolean types. For bool tensors, it computes the logical OR. 

Parameters
:   * **input** – the first input tensor
* **other** – the second input tensor

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.bitwise_or(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
tensor([-1, -2,  3], dtype=torch.int8)
>>> torch.bitwise_or(torch.tensor([True, True, False]), torch.tensor([False, True, False]))
tensor([ True, True, False])

```

