torch.nn.utils.parameters_to_vector 
============================================================================================================

torch.nn.utils. parameters_to_vector ( *parameters* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/utils/convert_parameters.py#L7) 
:   Flatten an iterable of parameters into a single vector. 

Parameters
: **parameters** ( *Iterable* *[* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *]*  ) – an iterable of Tensors that are the
parameters of a model.

Returns
:   The parameters represented by a single vector

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

