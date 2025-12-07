UnpackedDualTensor 
========================================================================

*class* torch.autograd.forward_ad. UnpackedDualTensor ( *primal*  , *tangent* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/autograd/forward_ad.py#L131) 
:   Namedtuple returned by [`unpack_dual()`](torch.autograd.forward_ad.unpack_dual.html#torch.autograd.forward_ad.unpack_dual "torch.autograd.forward_ad.unpack_dual")  containing the primal and tangent components of the dual tensor. 

See [`unpack_dual()`](torch.autograd.forward_ad.unpack_dual.html#torch.autograd.forward_ad.unpack_dual "torch.autograd.forward_ad.unpack_dual")  for more details. 

count ( *value*  , */* ) 
:   Return number of occurrences of value.

index ( *value*  , *start = 0*  , *stop = 9223372036854775807*  , */* ) 
:   Return first index of value. 

Raises ValueError if the value is not present.

primal *: [Tensor](../tensors.html#torch.Tensor "torch.Tensor")* 
:   Alias for field number 0

tangent *: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") [ [Tensor](../tensors.html#torch.Tensor "torch.Tensor") ]* 
:   Alias for field number 1

