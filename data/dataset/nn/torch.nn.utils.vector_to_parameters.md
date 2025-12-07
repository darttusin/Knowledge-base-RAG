torch.nn.utils.vector_to_parameters 
============================================================================================================

torch.nn.utils. vector_to_parameters ( *vec*  , *parameters* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/utils/convert_parameters.py#L29) 
:   Copy slices of a vector into an iterable of parameters. 

Parameters
:   * **vec** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – a single vector representing the parameters of a model.
* **parameters** ( *Iterable* *[* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *]*  ) – an iterable of Tensors that are the
parameters of a model.

