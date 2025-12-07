torch.nn.functional.scaled_dot_product_attention 
=======================================================================================================================================

torch.nn.functional. scaled_dot_product_attention ( ) 
:   scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
:   is_causal=False, scale=None, enable_gqa=False) -> Tensor:

Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed,
and applying dropout if a probability greater than 0.0 is specified. The optional scale argument can only be
specified as a keyword argument. 

```
# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

```

Warning 

This function is beta and subject to change.

Warning 

This function always applies dropout according to the specified `dropout_p`  argument.
To disable dropout during evaluation, be sure to pass a value of `0.0`  when the module
that makes the function call is not in training mode. 

For example: 

```
class MyModel(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, ...):
        return F.scaled_dot_product_attention(...,
            dropout_p=(self.p if self.training else 0.0))

```

Note 

There are currently three supported implementations of scaled dot product attention: 

> * [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
> * [Memory-Efficient Attention](https://github.com/facebookresearch/xformers)
> * A PyTorch implementation defined in C++ matching the above formulation

The function may call optimized kernels for improved performance when using the CUDA backend.
For all other backends, the PyTorch implementation will be used. 

All implementations are enabled by default. Scaled dot product attention attempts to automatically select the
most optimal implementation based on the inputs. In order to provide more fine-grained control over what implementation
is used, the following functions are provided for enabling and disabling implementations.
The context manager is the preferred mechanism: 

> * [`torch.nn.attention.sdpa_kernel()`](torch.nn.attention.sdpa_kernel.html#torch.nn.attention.sdpa_kernel "torch.nn.attention.sdpa_kernel")  : A context manager used to enable or disable any of the implementations.
> * [`torch.backends.cuda.enable_flash_sdp()`](../backends.html#torch.backends.cuda.enable_flash_sdp "torch.backends.cuda.enable_flash_sdp")  : Globally enables or disables FlashAttention.
> * [`torch.backends.cuda.enable_mem_efficient_sdp()`](../backends.html#torch.backends.cuda.enable_mem_efficient_sdp "torch.backends.cuda.enable_mem_efficient_sdp")  : Globally enables or disables Memory-Efficient Attention.
> * [`torch.backends.cuda.enable_math_sdp()`](../backends.html#torch.backends.cuda.enable_math_sdp "torch.backends.cuda.enable_math_sdp")  : Globally enables or disables the PyTorch C++ implementation.

Each of the fused kernels has specific input limitations. If the user requires the use of a specific fused implementation,
disable the PyTorch C++ implementation using [`torch.nn.attention.sdpa_kernel()`](torch.nn.attention.sdpa_kernel.html#torch.nn.attention.sdpa_kernel "torch.nn.attention.sdpa_kernel")  .
In the event that a fused implementation is not available, a warning will be raised with the
reasons why the fused implementation cannot run. 

Due to the nature of fusing floating point operations, the output of this function may be different
depending on what backend kernel is chosen.
The c++ implementation supports torch.float64 and can be used when higher precision is required.
For math backend, all intermediates are kept in torch.float if inputs are in torch.half or torch.bfloat16.

For more information please see [Numerical accuracy](../notes/numerical_accuracy.html) 

> Grouped Query Attention (GQA) is an experimental feature. It currently works only for Flash_attention
> and math kernel on CUDA tensor, and does not support Nested tensor.
> Constraints for GQA: 
> 
> 
> > * number_of_heads_query % number_of_heads_key_value == 0 and,
> > * number_of_heads_key == number_of_heads_value

Note 

In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting `torch.backends.cudnn.deterministic = True`  . See [Reproducibility](../notes/randomness.html)  for more information.

Parameters
:   * **query** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Query tensor; shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mi mathvariant="normal">
                .
               </mi>
<mi mathvariant="normal">
                .
               </mi>
<mi mathvariant="normal">
                .
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                H
               </mi>
<mi>
                q
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
               (N, ..., Hq, L, E)
              </annotation>
</semantics>
</math> -->( N , . . . , H q , L , E ) (N, ..., Hq, L, E)( N , ... , H q , L , E )  .

* **key** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Key tensor; shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mi mathvariant="normal">
                .
               </mi>
<mi mathvariant="normal">
                .
               </mi>
<mi mathvariant="normal">
                .
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                H
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
<mi>
                E
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, ..., H, S, E)
              </annotation>
</semantics>
</math> -->( N , . . . , H , S , E ) (N, ..., H, S, E)( N , ... , H , S , E )  .

* **value** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Value tensor; shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mi mathvariant="normal">
                .
               </mi>
<mi mathvariant="normal">
                .
               </mi>
<mi mathvariant="normal">
                .
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                H
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
<mi>
                E
               </mi>
<mi>
                v
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, ..., H, S, Ev)
              </annotation>
</semantics>
</math> -->( N , . . . , H , S , E v ) (N, ..., H, S, Ev)( N , ... , H , S , E v )  .

* **attn_mask** ( *optional Tensor*  ) – Attention mask; shape must be broadcastable to the shape of attention weights,
which is <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mi mathvariant="normal">
                .
               </mi>
<mi mathvariant="normal">
                .
               </mi>
<mi mathvariant="normal">
                .
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
               (N,..., L, S)
              </annotation>
</semantics>
</math> -->( N , . . . , L , S ) (N,..., L, S)( N , ... , L , S )  . Two types of masks are supported.
A boolean mask where a value of True indicates that the element *should*  take part in attention.
A float mask of the same type as query, key, value that is added to the attention score.

* **dropout_p** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – Dropout probability; if greater than 0.0, dropout is applied
* **is_causal** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If set to true, the attention masking is a lower triangular matrix when the mask is a
square matrix. The attention masking has the form of the upper left causal bias due to the alignment
(see [`torch.nn.attention.bias.CausalBias`](torch.nn.attention.bias.CausalBias.html#torch.nn.attention.bias.CausalBias "torch.nn.attention.bias.CausalBias")  ) when the mask is a non-square matrix.
An error is thrown if both attn_mask and is_causal are set.
* **scale** ( *optional python:float* *,* *keyword-only*  ) – Scaling factor applied prior to softmax. If None, the default value is set
to <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mfrac>
<mn>
                 1
                </mn>
<msqrt>
<mi>
                  E
                 </mi>
</msqrt>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
               frac{1}{sqrt{E}}
              </annotation>
</semantics>
</math> -->1 E frac{1}{sqrt{E}}E ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ 1 ​  .

* **enable_gqa** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If set to True, Grouped Query Attention (GQA) is enabled, by default it is set to False.

Returns
:   Attention output; shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mo separator="true">
              ,
             </mo>
<mi>
              H
             </mi>
<mi>
              q
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
<mi>
              v
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             (N, ..., Hq, L, Ev)
            </annotation>
</semantics>
</math> -->( N , . . . , H q , L , E v ) (N, ..., Hq, L, Ev)( N , ... , H q , L , E v )  .

Return type
:   output ( [Tensor](../tensors.html#torch.Tensor "torch.Tensor")  )

Shape legend:
:   * <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                N
               </mi>
<mo>
                :
               </mo>
<mtext>
                Batch size
               </mtext>
<mi mathvariant="normal">
                .
               </mi>
<mi mathvariant="normal">
                .
               </mi>
<mi mathvariant="normal">
                .
               </mi>
<mo>
                :
               </mo>
<mtext>
                Any number of other batch dimensions (optional)
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               N: text{Batch size} ... : text{Any number of other batch dimensions (optional)}
              </annotation>
</semantics>
</math> -->N : Batch size . . . : Any number of other batch dimensions (optional) N: text{Batch size} ... : text{Any number of other batch dimensions (optional)}N : Batch size ... : Any number of other batch dimensions (optional)

* <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                S
               </mi>
<mo>
                :
               </mo>
<mtext>
                Source sequence length
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               S: text{Source sequence length}
              </annotation>
</semantics>
</math> -->S : Source sequence length S: text{Source sequence length}S : Source sequence length

* <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                L
               </mi>
<mo>
                :
               </mo>
<mtext>
                Target sequence length
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               L: text{Target sequence length}
              </annotation>
</semantics>
</math> -->L : Target sequence length L: text{Target sequence length}L : Target sequence length

* <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                E
               </mi>
<mo>
                :
               </mo>
<mtext>
                Embedding dimension of the query and key
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               E: text{Embedding dimension of the query and key}
              </annotation>
</semantics>
</math> -->E : Embedding dimension of the query and key E: text{Embedding dimension of the query and key}E : Embedding dimension of the query and key

* <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                E
               </mi>
<mi>
                v
               </mi>
<mo>
                :
               </mo>
<mtext>
                Embedding dimension of the value
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               Ev: text{Embedding dimension of the value}
              </annotation>
</semantics>
</math> -->E v : Embedding dimension of the value Ev: text{Embedding dimension of the value}E v : Embedding dimension of the value

* <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                H
               </mi>
<mi>
                q
               </mi>
<mo>
                :
               </mo>
<mtext>
                Number of heads of query
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               Hq: text{Number of heads of query}
              </annotation>
</semantics>
</math> -->H q : Number of heads of query Hq: text{Number of heads of query}H q : Number of heads of query

* <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                H
               </mi>
<mo>
                :
               </mo>
<mtext>
                Number of heads of key and value
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               H: text{Number of heads of key and value}
              </annotation>
</semantics>
</math> -->H : Number of heads of key and value H: text{Number of heads of key and value}H : Number of heads of key and value

Examples 

```
>>> # Optionally use the context manager to ensure one of the fused kernels is run
>>> query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
>>> key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
>>> value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
>>> with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
>>>     F.scaled_dot_product_attention(query,key,value)

```

```
>>> # Sample for GQA for llama3
>>> query = torch.rand(32, 32, 128, 64, dtype=torch.float16, device="cuda")
>>> key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
>>> value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
>>> with sdpa_kernel(backends=[SDPBackend.MATH]):
>>>     F.scaled_dot_product_attention(query,key,value,enable_gqa=True)

```

