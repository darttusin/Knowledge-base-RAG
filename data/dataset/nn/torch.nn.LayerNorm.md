LayerNorm 
======================================================

*class* torch.nn. LayerNorm ( *normalized_shape*  , *eps = 1e-05*  , *elementwise_affine = True*  , *bias = True*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/normalization.py#L94) 
:   Applies Layer Normalization over a mini-batch of inputs. 

This layer implements the operation as described in
the paper [Layer Normalization](https://arxiv.org/abs/1607.06450) 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            y
           </mi>
<mo>
            =
           </mo>
<mfrac>
<mrow>
<mi>
              x
             </mi>
<mo>
              −
             </mo>
<mi mathvariant="normal">
              E
             </mi>
<mo stretchy="false">
              [
             </mo>
<mi>
              x
             </mi>
<mo stretchy="false">
              ]
             </mo>
</mrow>
<msqrt>
<mrow>
<mrow>
<mi mathvariant="normal">
                V
               </mi>
<mi mathvariant="normal">
                a
               </mi>
<mi mathvariant="normal">
                r
               </mi>
</mrow>
<mo stretchy="false">
               [
              </mo>
<mi>
               x
              </mi>
<mo stretchy="false">
               ]
              </mo>
<mo>
               +
              </mo>
<mi>
               ϵ
              </mi>
</mrow>
</msqrt>
</mfrac>
<mo>
            ∗
           </mo>
<mi>
            γ
           </mi>
<mo>
            +
           </mo>
<mi>
            β
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           y = frac{x - mathrm{E}[x]}{ sqrt{mathrm{Var}[x] + epsilon}} * gamma + beta
          </annotation>
</semantics>
</math> -->
y = x − E [ x ] V a r [ x ] + ϵ ∗ γ + β y = frac{x - mathrm{E}[x]}{ sqrt{mathrm{Var}[x] + epsilon}} * gamma + beta

y = Var [ x ] + ϵ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMjhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTI5NiIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNMjYzLDY4MWMwLjcsMCwxOCwzOS43LDUyLDExOQpjMzQsNzkuMyw2OC4xNjcsMTU4LjcsMTAyLjUsMjM4YzM0LjMsNzkuMyw1MS44LDExOS4zLDUyLjUsMTIwCmMzNDAsLTcwNC43LDUxMC43LC0xMDYwLjMsNTEyLC0xMDY3CmwwIC0wCmM0LjcsLTcuMywxMSwtMTEsMTksLTExCkg0MDAwMHY0MEgxMDEyLjMKcy0yNzEuMyw1NjcsLTI3MS4zLDU2N2MtMzguNyw4MC43LC04NCwxNzUsLTEzNiwyODNjLTUyLDEwOCwtODkuMTY3LDE4NS4zLC0xMTEuNSwyMzIKYy0yMi4zLDQ2LjcsLTMzLjgsNzAuMywtMzQuNSw3MWMtNC43LDQuNywtMTIuMyw3LC0yMyw3cy0xMiwtMSwtMTIsLTEKcy0xMDksLTI1MywtMTA5LC0yNTNjLTcyLjcsLTE2OCwtMTA5LjMsLTI1MiwtMTEwLC0yNTJjLTEwLjcsOCwtMjIsMTYuNywtMzQsMjYKYy0yMiwxNy4zLC0zMy4zLDI2LC0zNCwyNnMtMjYsLTI2LC0yNiwtMjZzNzYsLTU5LDc2LC01OXM3NiwtNjAsNzYsLTYwegpNMTAwMSA4MGg0MDAwMDB2NDBoLTQwMDAwMHoiPgo8L3BhdGg+Cjwvc3ZnPg==)​ x − E [ x ] ​ ∗ γ + β

The mean and standard-deviation are calculated over the last *D* dimensions, where *D* is the dimension of `normalized_shape`  . For example, if `normalized_shape`  is `(3, 5)`  (a 2-dimensional shape), the mean and standard-deviation are computed over
the last 2 dimensions of the input (i.e. `input.mean((-2, -1))`  ). <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            γ
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           gamma
          </annotation>
</semantics>
</math> -->γ gammaγ  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            β
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           beta
          </annotation>
</semantics>
</math> -->β betaβ  are learnable affine transform parameters of `normalized_shape`  if `elementwise_affine`  is `True`  .
The variance is calculated via the biased estimator, equivalent to *torch.var(input, unbiased=False)* . 

Note 

Unlike Batch Normalization and Instance Normalization, which applies
scalar scale and bias for each entire channel/plane with the `affine`  option, Layer Normalization applies per-element scale and
bias with `elementwise_affine`  .

This layer uses statistics computed from input data in both training and
evaluation modes. 

Parameters
:   * **normalized_shape** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)") *or* [*torch.Size*](../size.html#torch.Size "torch.Size")  ) –

    input shape from an expected input
        of size

    <!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
    <semantics>
    <mrow>
    <mo stretchy="false">
                    [
                   </mo>
    <mo>
                    ∗
                   </mo>
    <mo>
                    ×
                   </mo>
    <mtext>
                    normalized_shape
                   </mtext>
    <mo stretchy="false">
                    [
                   </mo>
    <mn>
                    0
                   </mn>
    <mo stretchy="false">
                    ]
                   </mo>
    <mo>
                    ×
                   </mo>
    <mtext>
                    normalized_shape
                   </mtext>
    <mo stretchy="false">
                    [
                   </mo>
    <mn>
                    1
                   </mn>
    <mo stretchy="false">
                    ]
                   </mo>
    <mo>
                    ×
                   </mo>
    <mo>
                    …
                   </mo>
    <mo>
                    ×
                   </mo>
    <mtext>
                    normalized_shape
                   </mtext>
    <mo stretchy="false">
                    [
                   </mo>
    <mo>
                    −
                   </mo>
    <mn>
                    1
                   </mn>
    <mo stretchy="false">
                    ]
                   </mo>
    <mo stretchy="false">
                    ]
                   </mo>
    </mrow>
    <annotation encoding="application/x-tex">
                   [* times text{normalized_shape}[0] times text{normalized_shape}[1]
        times ldots times text{normalized_shape}[-1]]
                  </annotation>
    </semantics>
    </math> -->
    [ ∗ × normalized_shape [ 0 ] × normalized_shape [ 1 ] × … × normalized_shape [ − 1 ] ] [* times text{normalized_shape}[0] times text{normalized_shape}[1]
     times ldots times text{normalized_shape}[-1]]

    [ ∗ × normalized_shape [ 0 ] × normalized_shape [ 1 ] × … × normalized_shape [ − 1 ]]

    If a single integer is used, it is treated as a singleton list, and this module will
        normalize over the last dimension which is expected to be of that specific size.

* **eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – a value added to the denominator for numerical stability. Default: 1e-5
* **elementwise_affine** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – a boolean value that when set to `True`  , this module
has learnable per-element affine parameters initialized to ones (for weights)
and zeros (for biases). Default: `True`  .
* **bias** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If set to `False`  , the layer will not learn an additive bias (only relevant if `elementwise_affine`  is `True`  ). Default: `True`  .

Variables
:   * **weight** – the learnable weights of the module of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
                normalized_shape
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               text{normalized_shape}
              </annotation>
</semantics>
</math> -->normalized_shape text{normalized_shape}normalized_shape  when `elementwise_affine`  is set to `True`  .
The values are initialized to 1.

* **bias** – the learnable bias of the module of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
                normalized_shape
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               text{normalized_shape}
              </annotation>
</semantics>
</math> -->normalized_shape text{normalized_shape}normalized_shape  when `elementwise_affine`  is set to `True`  .
The values are initialized to 0.

Shape:
:   * Input: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mo>
                ∗
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, *)
              </annotation>
</semantics>
</math> -->( N , ∗ ) (N, *)( N , ∗ )

* Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mo>
                ∗
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, *)
              </annotation>
</semantics>
</math> -->( N , ∗ ) (N, *)( N , ∗ )  (same shape as input)

Examples: 

```
>>> # NLP Example
>>> batch, sentence_length, embedding_dim = 20, 5, 10
>>> embedding = torch.randn(batch, sentence_length, embedding_dim)
>>> layer_norm = nn.LayerNorm(embedding_dim)
>>> # Activate module
>>> layer_norm(embedding)
>>>
>>> # Image Example
>>> N, C, H, W = 20, 5, 10, 10
>>> input = torch.randn(N, C, H, W)
>>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
>>> # as shown in the image below
>>> layer_norm = nn.LayerNorm([C, H, W])
>>> output = layer_norm(input)

```

[![../_images/layer_norm.jpg](../_images/layer_norm.jpg)](../_images/layer_norm.jpg)

