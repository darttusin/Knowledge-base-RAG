torch.Tensor.expand_as 
=================================================================================

Tensor. expand_as ( *other* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Expand this tensor to the same size as `other`  . `self.expand_as(other)`  is equivalent to `self.expand(other.size())`  . 

Please see [`expand()`](torch.Tensor.expand.html#torch.Tensor.expand "torch.Tensor.expand")  for more information about `expand`  . 

Parameters
: **other** ( [`torch.Tensor`](../tensors.html#torch.Tensor "torch.Tensor")  ) – The result tensor has the same size
as `other`  .

