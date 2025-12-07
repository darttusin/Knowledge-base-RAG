torch.nn.utils.fuse_linear_bn_weights 
=================================================================================================================

torch.nn.utils. fuse_linear_bn_weights ( *linear_w*  , *linear_b*  , *bn_rm*  , *bn_rv*  , *bn_eps*  , *bn_w*  , *bn_b* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/utils/fusion.py#L156) 
:   Fuse linear module parameters and BatchNorm module parameters into new linear module parameters. 

Parameters
:   * **linear_w** ( [*torch.Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Linear weight.
* **linear_b** ( *Optional* *[* [*torch.Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *]*  ) – Linear bias.
* **bn_rm** ( [*torch.Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – BatchNorm running mean.
* **bn_rv** ( [*torch.Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – BatchNorm running variance.
* **bn_eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – BatchNorm epsilon.
* **bn_w** ( [*torch.Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – BatchNorm weight.
* **bn_b** ( [*torch.Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – BatchNorm bias.

Returns
:   Fused linear weight and bias.

Return type
:   Tuple[torch.nn.Parameter, torch.nn.Parameter]

