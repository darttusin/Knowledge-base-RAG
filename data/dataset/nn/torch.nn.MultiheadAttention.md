MultiheadAttention 
========================================================================

*class* torch.nn. MultiheadAttention ( *embed_dim*  , *num_heads*  , *dropout = 0.0*  , *bias = True*  , *add_bias_kv = False*  , *add_zero_attn = False*  , *kdim = None*  , *vdim = None*  , *batch_first = False*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L977) 
:   Allows the model to jointly attend to information from different representation subspaces. 

This MultiheadAttention layer implements the original architecture described
in the [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  paper. The
intent of this layer is as a reference implementation for foundational understanding
and thus it contains only limited features relative to newer architectures.
Given the fast pace of innovation in transformer-like architectures, we recommend
exploring this [tutorial](https://localhost:8000/tutorials/intermediate/transformer_building_blocks.html)  to build efficient layers from building blocks in core or using higher
level libraries from the [PyTorch Ecosystem](https://landscape.localhost:8000/)  . 

Multi-Head Attention is defined as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            MultiHead
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mi>
            Q
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            K
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            V
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mtext>
            Concat
           </mtext>
<mo stretchy="false">
            (
           </mo>
<msub>
<mtext>
             head
            </mtext>
<mn>
             1
            </mn>
</msub>
<mo separator="true">
            ,
           </mo>
<mo>
            …
           </mo>
<mo separator="true">
            ,
           </mo>
<msub>
<mtext>
             head
            </mtext>
<mi>
             h
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
<msup>
<mi>
             W
            </mi>
<mi>
             O
            </mi>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           text{MultiHead}(Q, K, V) = text{Concat}(text{head}_1,dots,text{head}_h)W^O
          </annotation>
</semantics>
</math> -->
MultiHead ( Q , K , V ) = Concat ( head 1 , … , head h ) W O text{MultiHead}(Q, K, V) = text{Concat}(text{head}_1,dots,text{head}_h)W^O

MultiHead ( Q , K , V ) = Concat ( head 1 ​ , … , head h ​ ) W O

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mtext>
             head
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo>
            =
           </mo>
<mtext>
            Attention
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mi>
            Q
           </mi>
<msubsup>
<mi>
             W
            </mi>
<mi>
             i
            </mi>
<mi>
             Q
            </mi>
</msubsup>
<mo separator="true">
            ,
           </mo>
<mi>
            K
           </mi>
<msubsup>
<mi>
             W
            </mi>
<mi>
             i
            </mi>
<mi>
             K
            </mi>
</msubsup>
<mo separator="true">
            ,
           </mo>
<mi>
            V
           </mi>
<msubsup>
<mi>
             W
            </mi>
<mi>
             i
            </mi>
<mi>
             V
            </mi>
</msubsup>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{head}_i = text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
          </annotation>
</semantics>
</math> -->head i = Attention ( Q W i Q , K W i K , V W i V ) text{head}_i = text{Attention}(QW_i^Q, KW_i^K, VW_i^V)head i ​ = Attention ( Q W i Q ​ , K W i K ​ , V W i V ​ )  . 

`nn.MultiheadAttention`  will use the optimized implementations of `scaled_dot_product_attention()`  when possible. 

In addition to support for the new `scaled_dot_product_attention()`  function, for speeding up Inference, MHA will use
fastpath inference with support for Nested Tensors, iff: 

* self attention is being computed (i.e., `query`  , `key`  , and `value`  are the same tensor).
* inputs are batched (3D) with `batch_first==True`
* Either autograd is disabled (using `torch.inference_mode`  or `torch.no_grad`  ) or no tensor argument `requires_grad`
* training is disabled (using `.eval()`  )
* `add_bias_kv`  is `False`
* `add_zero_attn`  is `False`
* `kdim`  and `vdim`  are equal to `embed_dim`
* if a [NestedTensor](https://localhost:8000/docs/stable/nested.html)  is passed, neither `key_padding_mask`  nor `attn_mask`  is passed
* autocast is disabled

If the optimized inference fastpath implementation is in use, a [NestedTensor](https://localhost:8000/docs/stable/nested.html)  can be passed for `query`  / `key`  / `value`  to represent padding more efficiently than using a
padding mask. In this case, a [NestedTensor](https://localhost:8000/docs/stable/nested.html)  will be returned, and an additional speedup proportional to the fraction of the input
that is padding can be expected. 

Parameters
:   * **embed_dim** – Total dimension of the model.
* **num_heads** – Number of parallel attention heads. Note that `embed_dim`  will be split
across `num_heads`  (i.e. each head will have dimension `embed_dim // num_heads`  ).
* **dropout** – Dropout probability on `attn_output_weights`  . Default: `0.0`  (no dropout).
* **bias** – If specified, adds bias to input / output projection layers. Default: `True`  .
* **add_bias_kv** – If specified, adds bias to the key and value sequences at dim=0. Default: `False`  .
* **add_zero_attn** – If specified, adds a new batch of zeros to the key and value sequences at dim=1.
Default: `False`  .
* **kdim** – Total number of features for keys. Default: `None`  (uses `kdim=embed_dim`  ).
* **vdim** – Total number of features for values. Default: `None`  (uses `vdim=embed_dim`  ).
* **batch_first** – If `True`  , then the input and output tensors are provided
as (batch, seq, feature). Default: `False`  (seq, batch, feature).

Examples: 

```
>>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
>>> attn_output, attn_output_weights = multihead_attn(query, key, value)

```

forward ( *query*  , *key*  , *value*  , *key_padding_mask = None*  , *need_weights = True*  , *attn_mask = None*  , *average_attn_weights = True*  , *is_causal = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L1144) 
:   Compute attention outputs using query, key, and value embeddings. 

> Supports optional parameters for padding, masks and attention weights.

Parameters
:   * **query** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Query embeddings of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  L
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<msub>
<mi>
                   E
                  </mi>
<mi>
                   q
                  </mi>
</msub>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (L, E_q)
                </annotation>
</semantics>
</math> -->( L , E q ) (L, E_q)( L , E q ​ )  for unbatched input, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  L
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
<msub>
<mi>
                   E
                  </mi>
<mi>
                   q
                  </mi>
</msub>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (L, N, E_q)
                </annotation>
</semantics>
</math> -->( L , N , E q ) (L, N, E_q)( L , N , E q ​ )  when `batch_first=False`  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                  L
                 </mi>
<mo separator="true">
                  ,
                 </mo>
<msub>
<mi>
                   E
                  </mi>
<mi>
                   q
                  </mi>
</msub>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (N, L, E_q)
                </annotation>
</semantics>
</math> -->( N , L , E q ) (N, L, E_q)( N , L , E q ​ )  when `batch_first=True`  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                  L
                 </mi>
</mrow>
<annotation encoding="application/x-tex">
                 L
                </annotation>
</semantics>
</math> -->L LL  is the target sequence length, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->N NN  is the batch size, and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
                   E
                  </mi>
<mi>
                   q
                  </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
                 E_q
                </annotation>
</semantics>
</math> -->E q E_qE q ​  is the query embedding dimension `embed_dim`  .
Queries are compared against key-value pairs to produce the output.
See “Attention Is All You Need” for more details.

* **key** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Key embeddings of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<msub>
<mi>
                   E
                  </mi>
<mi>
                   k
                  </mi>
</msub>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (S, E_k)
                </annotation>
</semantics>
</math> -->( S , E k ) (S, E_k)( S , E k ​ )  for unbatched input, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<msub>
<mi>
                   E
                  </mi>
<mi>
                   k
                  </mi>
</msub>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (S, N, E_k)
                </annotation>
</semantics>
</math> -->( S , N , E k ) (S, N, E_k)( S , N , E k ​ )  when `batch_first=False`  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mo separator="true">
                  ,
                 </mo>
<msub>
<mi>
                   E
                  </mi>
<mi>
                   k
                  </mi>
</msub>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (N, S, E_k)
                </annotation>
</semantics>
</math> -->( N , S , E k ) (N, S, E_k)( N , S , E k ​ )  when `batch_first=True`  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                  N
                 </mi>
</mrow>
<annotation encoding="application/x-tex">
                 N
                </annotation>
</semantics>
</math> -->N NN  is the batch size, and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
                   E
                  </mi>
<mi>
                   k
                  </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
                 E_k
                </annotation>
</semantics>
</math> -->E k E_kE k ​  is the key embedding dimension `kdim`  .
See “Attention Is All You Need” for more details.

* **value** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Value embeddings of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<msub>
<mi>
                   E
                  </mi>
<mi>
                   v
                  </mi>
</msub>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (S, E_v)
                </annotation>
</semantics>
</math> -->( S , E v ) (S, E_v)( S , E v ​ )  for unbatched input, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<msub>
<mi>
                   E
                  </mi>
<mi>
                   v
                  </mi>
</msub>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (S, N, E_v)
                </annotation>
</semantics>
</math> -->( S , N , E v ) (S, N, E_v)( S , N , E v ​ )  when `batch_first=False`  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mo separator="true">
                  ,
                 </mo>
<msub>
<mi>
                   E
                  </mi>
<mi>
                   v
                  </mi>
</msub>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<annotation encoding="application/x-tex">
                 (N, S, E_v)
                </annotation>
</semantics>
</math> -->( N , S , E v ) (N, S, E_v)( N , S , E v ​ )  when `batch_first=True`  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->S SS  is the source
sequence length, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->N NN  is the batch size, and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
                   E
                  </mi>
<mi>
                   v
                  </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
                 E_v
                </annotation>
</semantics>
</math> -->E v E_vE v ​  is the value embedding dimension `vdim`  .
See “Attention Is All You Need” for more details.

* **key_padding_mask** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *]*  ) – If specified, a mask of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N , S ) (N, S)( N , S )  indicating which elements within `key`  to ignore for the purpose of attention (i.e. treat as “padding”). For unbatched *query* , shape should be <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( S ) (S)( S )  .
Binary and float masks are supported.
For a binary mask, a `True`  value indicates that the corresponding `key`  value will be ignored for
the purpose of attention. For a float mask, it will be directly added to the corresponding `key`  value.

* **need_weights** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If specified, returns `attn_output_weights`  in addition to `attn_outputs`  .
Set `need_weights=False`  to use the optimized `scaled_dot_product_attention`  and achieve the best performance for MHA.
Default: `True`  .
* **attn_mask** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *]*  ) – If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  L
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
                 (L, S)
                </annotation>
</semantics>
</math> -->( L , S ) (L, S)( L , S )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                  L
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
                 (Ncdottext{num_heads}, L, S)
                </annotation>
</semantics>
</math> -->( N ⋅ num_heads , L , S ) (Ncdottext{num_heads}, L, S)( N ⋅ num_heads , L , S )  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->N NN  is the batch size, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                  L
                 </mi>
</mrow>
<annotation encoding="application/x-tex">
                 L
                </annotation>
</semantics>
</math> -->L LL  is the target sequence length, and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->S SS  is the source sequence length. A 2D mask will be
broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
Binary and float masks are supported. For a binary mask, a `True`  value indicates that the
corresponding position is not allowed to attend. For a float mask, the mask values will be added to
the attention weight.
If both attn_mask and key_padding_mask are supplied, their types should match.

* **average_attn_weights** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If true, indicates that the returned `attn_weights`  should be averaged across
heads. Otherwise, `attn_weights`  are provided separately per head. Note that this flag only has an
effect when `need_weights=True`  . Default: `True`  (i.e. average weights across heads)
* **is_causal** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If specified, applies a causal mask as attention mask.
Default: `False`  .
Warning: `is_causal`  provides a hint that `attn_mask`  is the
causal mask. Providing incorrect hints can result in
incorrect execution, including forward and backward
compatibility.

Return type
:   [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [ [torch.Tensor](../tensors.html#torch.Tensor "torch.Tensor")  , [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)")  [ [torch.Tensor](../tensors.html#torch.Tensor "torch.Tensor")  ]]

Outputs:
:   * **attn_output** - Attention outputs of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  L
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
                 (L, E)
                </annotation>
</semantics>
</math> -->( L , E ) (L, E)( L , E )  when input is unbatched, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  L
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
                 (L, N, E)
                </annotation>
</semantics>
</math> -->( L , N , E ) (L, N, E)( L , N , E )  when `batch_first=False`  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                  L
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
                 (N, L, E)
                </annotation>
</semantics>
</math> -->( N , L , E ) (N, L, E)( N , L , E )  when `batch_first=True`  ,
where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                  L
                 </mi>
</mrow>
<annotation encoding="application/x-tex">
                 L
                </annotation>
</semantics>
</math> -->L LL  is the target sequence length, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->N NN  is the batch size, and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->E EE  is the
embedding dimension `embed_dim`  .

* **attn_output_weights** - Only returned when `need_weights=True`  . If `average_attn_weights=True`  ,
returns attention weights averaged across heads of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  L
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
                 (L, S)
                </annotation>
</semantics>
</math> -->( L , S ) (L, S)( L , S )  when input is unbatched or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                  L
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
                 (N, L, S)
                </annotation>
</semantics>
</math> -->( N , L , S ) (N, L, S)( N , L , S )  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->N NN  is the batch size, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                  L
                 </mi>
</mrow>
<annotation encoding="application/x-tex">
                 L
                </annotation>
</semantics>
</math> -->L LL  is the target sequence length, and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->S SS  is the source sequence length. If `average_attn_weights=False`  , returns attention weights per
head of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                  (
                 </mo>
<mtext>
                  num_heads
                 </mtext>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  L
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
                 (text{num_heads}, L, S)
                </annotation>
</semantics>
</math> -->( num_heads , L , S ) (text{num_heads}, L, S)( num_heads , L , S )  when input is unbatched or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mtext>
                  num_heads
                 </mtext>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  L
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
                 (N, text{num_heads}, L, S)
                </annotation>
</semantics>
</math> -->( N , num_heads , L , S ) (N, text{num_heads}, L, S)( N , num_heads , L , S )  .

Note 

*batch_first* argument is ignored for unbatched inputs.

merge_masks ( *attn_mask*  , *key_padding_mask*  , *query* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L1406) 
:   Determine mask type and combine masks if necessary. 

If only one mask is provided, that mask
and the corresponding mask type will be returned. If both masks are provided, they will be both
expanded to shape `(batch_size, num_heads, seq_len, seq_len)`  , combined with logical `or`  and mask type 2 will be returned
:param attn_mask: attention mask of shape `(seq_len, seq_len)`  , mask type 0
:param key_padding_mask: padding mask of shape `(batch_size, seq_len)`  , mask type 1
:param query: query embeddings of shape `(batch_size, seq_len, embed_dim)` 

Returns
:   merged mask
mask_type: merged mask type (0, 1, or 2)

Return type
:   merged_mask

