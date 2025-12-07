torch.clone 
==========================================================

torch. clone ( *input*  , *** , *memory_format = torch.preserve_format* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a copy of `input`  . 

Note 

This function is differentiable, so gradients will flow back from the
result of this operation to `input`  . To create a tensor without an
autograd relationship to `input`  see [`detach()`](torch.Tensor.detach.html#torch.Tensor.detach "torch.Tensor.detach")  .

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **memory_format** ( [`torch.memory_format`](../tensor_attributes.html#torch.memory_format "torch.memory_format")  , optional) – the desired memory format of
returned tensor. Default: `torch.preserve_format`  .

