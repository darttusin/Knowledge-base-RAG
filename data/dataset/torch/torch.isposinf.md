torch.isposinf 
================================================================

torch. isposinf ( *input*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Tests if each element of `input`  is positive infinity or not. 

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.tensor([-float('inf'), float('inf'), 1.2])
>>> torch.isposinf(a)
tensor([False,  True, False])

```

