Transformer 
==========================================================

*class* torch.nn. Transformer ( *d_model=512*  , *nhead=8*  , *num_encoder_layers=6*  , *num_decoder_layers=6*  , *dim_feedforward=2048*  , *dropout=0.1*  , *activation=<function relu>*  , *custom_encoder=None*  , *custom_decoder=None*  , *layer_norm_eps=1e-05*  , *batch_first=False*  , *norm_first=False*  , *bias=True*  , *device=None*  , *dtype=None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/transformer.py#L57) 
:   A basic transformer layer. 

This Transformer layer implements the original Transformer architecture described
in the [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  paper. The
intent of this layer is as a reference implementation for foundational understanding
and thus it contains only limited features relative to newer Transformer architectures.
Given the fast pace of innovation in transformer-like architectures, we recommend
exploring this [tutorial](https://localhost:8000/tutorials/intermediate/transformer_building_blocks.html)  to build an efficient transformer layer from building blocks in core or using higher
level libraries from the [PyTorch Ecosystem](https://landscape.localhost:8000/)  . 

Parameters
:   * **d_model** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the number of expected features in the encoder/decoder inputs (default=512).
* **nhead** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the number of heads in the multiheadattention models (default=8).
* **num_encoder_layers** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the number of sub-encoder-layers in the encoder (default=6).
* **num_decoder_layers** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the number of sub-decoder-layers in the decoder (default=6).
* **dim_feedforward** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the dimension of the feedforward network model (default=2048).
* **dropout** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the dropout value (default=0.1).
* **activation** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* [*Callable*](https://docs.python.org/3/library/typing.html#typing.Callable "(in Python v3.13)") *[* *[* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *]* *,* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *]* *]*  ) – the activation function of encoder/decoder intermediate layer, can be a string
(“relu” or “gelu”) or a unary callable. Default: relu
* **custom_encoder** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)") *]*  ) – custom encoder (default=None).
* **custom_decoder** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)") *]*  ) – custom decoder (default=None).
* **layer_norm_eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the eps value in layer normalization components (default=1e-5).
* **batch_first** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If `True`  , then the input and output tensors are provided
as (batch, seq, feature). Default: `False`  (seq, batch, feature).
* **norm_first** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – if `True`  , encoder and decoder layers will perform LayerNorms before
other attention and feedforward operations, otherwise after. Default: `False`  (after).
* **bias** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If set to `False`  , `Linear`  and `LayerNorm`  layers will not learn an additive
bias. Default: `True`  .

Examples 

```
>>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
>>> src = torch.rand((10, 32, 512))
>>> tgt = torch.rand((20, 32, 512))
>>> out = transformer_model(src, tgt)

```

Note: A full example to apply nn.Transformer module for the word language model is available in [pytorch/examples](https://github.com/pytorch/examples/tree/master/word_language_model) 

forward ( *src*  , *tgt*  , *src_mask = None*  , *tgt_mask = None*  , *memory_mask = None*  , *src_key_padding_mask = None*  , *tgt_key_padding_mask = None*  , *memory_key_padding_mask = None*  , *src_is_causal = None*  , *tgt_is_causal = None*  , *memory_is_causal = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/transformer.py#L172) 
:   Take in and process masked source/target sequences. 

Note 

If a boolean tensor is provided for any of the [src/tgt/memory]_mask arguments, positions with a `True`  value are
not allowed to participate in the attention,
which is the opposite of the definition for `attn_mask`  in [`torch.nn.functional.scaled_dot_product_attention()`](torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention "torch.nn.functional.scaled_dot_product_attention")  .

Parameters
:   * **src** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the sequence to the encoder (required).
* **tgt** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the sequence to the decoder (required).
* **src_mask** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *]*  ) – the additive mask for the src sequence (optional).
* **tgt_mask** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *]*  ) – the additive mask for the tgt sequence (optional).
* **memory_mask** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *]*  ) – the additive mask for the encoder output (optional).
* **src_key_padding_mask** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *]*  ) – the Tensor mask for src keys per batch (optional).
* **tgt_key_padding_mask** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *]*  ) – the Tensor mask for tgt keys per batch (optional).
* **memory_key_padding_mask** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *]*  ) – the Tensor mask for memory keys per batch (optional).
* **src_is_causal** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *]*  ) – If specified, applies a causal mask as `src_mask`  .
Default: `None`  ; try to detect a causal mask.
Warning: `src_is_causal`  provides a hint that `src_mask`  is
the causal mask. Providing incorrect hints can result in
incorrect execution, including forward and backward
compatibility.
* **tgt_is_causal** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *]*  ) – If specified, applies a causal mask as `tgt_mask`  .
Default: `None`  ; try to detect a causal mask.
Warning: `tgt_is_causal`  provides a hint that `tgt_mask`  is
the causal mask. Providing incorrect hints can result in
incorrect execution, including forward and backward
compatibility.
* **memory_is_causal** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If specified, applies a causal mask as `memory_mask`  .
Default: `False`  .
Warning: `memory_is_causal`  provides a hint that `memory_mask`  is the causal mask. Providing incorrect
hints can result in incorrect execution, including
forward and backward compatibility.

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

Shape:
:   * src: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  S
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  E
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (S, E)
                </annotation>
</semantics>
</math> -->( S , E ) (S, E)( S , E )  for unbatched input, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  S
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  N
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  E
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (S, N, E)
                </annotation>
</semantics>
</math> -->( S , N , E ) (S, N, E)( S , N , E )  if *batch_first=False* or *(N, S, E)* if *batch_first=True* .

* tgt: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  T
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  E
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (T, E)
                </annotation>
</semantics>
</math> -->( T , E ) (T, E)( T , E )  for unbatched input, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  T
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  N
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  E
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (T, N, E)
                </annotation>
</semantics>
</math> -->( T , N , E ) (T, N, E)( T , N , E )  if *batch_first=False* or *(N, T, E)* if *batch_first=True* .

* src_mask: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  S
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  S
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (S, S)
                </annotation>
</semantics>
</math> -->( S , S ) (S, S)( S , S )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  N
                 </mi>
<mo>
                  ⋅
                 </mo>
<mtext>
                  num_heads
                 </mtext>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  S
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  S
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (Ncdottext{num_heads}, S, S)
                </annotation>
</semantics>
</math> -->( N ⋅ num_heads , S , S ) (Ncdottext{num_heads}, S, S)( N ⋅ num_heads , S , S )  .

* tgt_mask: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  T
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  T
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (T, T)
                </annotation>
</semantics>
</math> -->( T , T ) (T, T)( T , T )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  N
                 </mi>
<mo>
                  ⋅
                 </mo>
<mtext>
                  num_heads
                 </mtext>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  T
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  T
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (Ncdottext{num_heads}, T, T)
                </annotation>
</semantics>
</math> -->( N ⋅ num_heads , T , T ) (Ncdottext{num_heads}, T, T)( N ⋅ num_heads , T , T )  .

* memory_mask: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  T
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  S
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (T, S)
                </annotation>
</semantics>
</math> -->( T , S ) (T, S)( T , S )  .

* src_key_padding_mask: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  S
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (S)
                </annotation>
</semantics>
</math> -->( S ) (S)( S )  for unbatched input otherwise <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  N
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  S
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (N, S)
                </annotation>
</semantics>
</math> -->( N , S ) (N, S)( N , S )  .

* tgt_key_padding_mask: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  T
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (T)
                </annotation>
</semantics>
</math> -->( T ) (T)( T )  for unbatched input otherwise <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  N
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  T
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (N, T)
                </annotation>
</semantics>
</math> -->( N , T ) (N, T)( N , T )  .

* memory_key_padding_mask: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  S
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (S)
                </annotation>
</semantics>
</math> -->( S ) (S)( S )  for unbatched input otherwise <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  N
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  S
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (N, S)
                </annotation>
</semantics>
</math> -->( N , S ) (N, S)( N , S )  .

Note: [src/tgt/memory]_mask ensures that position <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                i
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               i
              </annotation>
</semantics>
</math> -->i ii  is allowed to attend the unmasked
positions. If a BoolTensor is provided, positions with `True`  are not allowed to attend while `False`  values will be unchanged. If a FloatTensor
is provided, it will be added to the attention weight.
[src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
the attention. If a BoolTensor is provided, the positions with the
value of `True`  will be ignored while the position with the value of `False`  will be unchanged. 

* output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  T
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  E
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (T, E)
                </annotation>
</semantics>
</math> -->( T , E ) (T, E)( T , E )  for unbatched input, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  T
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  N
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  E
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (T, N, E)
                </annotation>
</semantics>
</math> -->( T , N , E ) (T, N, E)( T , N , E )  if *batch_first=False* or *(N, T, E)* if *batch_first=True* .

Note: Due to the multi-head attention architecture in the transformer model,
the output sequence length of a transformer is same as the input sequence
(i.e. target) length of the decoder. 

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                S
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               S
              </annotation>
</semantics>
</math> -->S SS  is the source sequence length, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                T
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               T
              </annotation>
</semantics>
</math> -->T TT  is the target sequence length, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                N
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               N
              </annotation>
</semantics>
</math> -->N NN  is the
batch size, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                E
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               E
              </annotation>
</semantics>
</math> -->E EE  is the feature number

Examples 

```
>>> output = transformer_model(
...     src, tgt, src_mask=src_mask, tgt_mask=tgt_mask
... )

```

*static* generate_square_subsequent_mask ( *sz*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/transformer.py#L292) 
:   Generate a square causal mask for the sequence. 

The masked positions are filled with float(‘-inf’). Unmasked positions are filled with float(0.0). 

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

