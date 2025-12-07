TransformerEncoder 
========================================================================

*class* torch.nn. TransformerEncoder ( *encoder_layer*  , *num_layers*  , *norm = None*  , *enable_nested_tensor = True*  , *mask_check = True* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/transformer.py#L311) 
:   TransformerEncoder is a stack of N encoder layers. 

This TransformerEncoder layer implements the original architecture described
in the [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  paper. The
intent of this layer is as a reference implementation for foundational understanding
and thus it contains only limited features relative to newer Transformer architectures.
Given the fast pace of innovation in transformer-like architectures, we recommend
exploring this [tutorial](https://localhost:8000/tutorials/intermediate/transformer_building_blocks.html)  to build efficient layers from building blocks in core or using higher
level libraries from the [PyTorch Ecosystem](https://landscape.localhost:8000/)  . 

Warning 

All layers in the TransformerEncoder are initialized with the same parameters.
It is recommended to manually initialize the layers after creating the TransformerEncoder instance.

Parameters
:   * **encoder_layer** ( [*TransformerEncoderLayer*](torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer "torch.nn.TransformerEncoderLayer")  ) – an instance of the TransformerEncoderLayer() class (required).
* **num_layers** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the number of sub-encoder-layers in the encoder (required).
* **norm** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Module*](torch.nn.Module.html#torch.nn.Module "torch.nn.modules.module.Module") *]*  ) – the layer normalization component (optional).
* **enable_nested_tensor** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – if True, input will automatically convert to nested tensor
(and convert back on output). This will improve the overall performance of
TransformerEncoder when padding rate is high. Default: `True`  (enabled).

Examples 

```
>>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
>>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
>>> src = torch.rand(10, 32, 512)
>>> out = transformer_encoder(src)

```

forward ( *src*  , *mask = None*  , *src_key_padding_mask = None*  , *is_causal = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/transformer.py#L397) 
:   Pass the input through the encoder layers in turn. 

Parameters
:   * **src** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the sequence to the encoder (required).
* **mask** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *]*  ) – the mask for the src sequence (optional).
* **src_key_padding_mask** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *]*  ) – the mask for the src keys per batch (optional).
* **is_causal** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *]*  ) – If specified, applies a causal mask as `mask`  .
Default: `None`  ; try to detect a causal mask.
Warning: `is_causal`  provides a hint that `mask`  is the
causal mask. Providing incorrect hints can result in
incorrect execution, including forward and backward
compatibility.

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

Shape:
:   see the docs in [`Transformer`](torch.nn.Transformer.html#torch.nn.Transformer "torch.nn.Transformer")  .

