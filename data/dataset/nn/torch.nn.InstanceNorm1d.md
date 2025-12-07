InstanceNorm1d 
================================================================

*class* torch.nn. InstanceNorm1d ( *num_features*  , *eps = 1e-05*  , *momentum = 0.1*  , *affine = False*  , *track_running_stats = False*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/instancenorm.py#L127) 
:   Applies Instance Normalization. 

This operation applies Instance Normalization
over a 2D (unbatched) or 3D (batched) input as described in the paper [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)  . 

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

The mean and standard-deviation are calculated per-dimension separately
for each object in a mini-batch. <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->β betaβ  are learnable parameter vectors
of size *C* (where *C* is the number of features or channels of the input) if `affine`  is `True`  .
The variance is calculated via the biased estimator, equivalent to *torch.var(input, unbiased=False)* . 

By default, this layer uses instance statistics computed from input data in
both training and evaluation modes. 

If `track_running_stats`  is set to `True`  , during training this
layer keeps running estimates of its computed mean and variance, which are
then used for normalization during evaluation. The running estimates are
kept with a default `momentum`  of 0.1. 

Note 

This `momentum`  argument is different from one used in optimizer
classes and the conventional notion of momentum. Mathematically, the
update rule for running statistics here is <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mover accent="true">
<mi>
               x
              </mi>
<mo>
               ^
              </mo>
</mover>
<mtext>
              new
             </mtext>
</msub>
<mo>
             =
            </mo>
<mo stretchy="false">
             (
            </mo>
<mn>
             1
            </mn>
<mo>
             −
            </mo>
<mtext>
             momentum
            </mtext>
<mo stretchy="false">
             )
            </mo>
<mo>
             ×
            </mo>
<mover accent="true">
<mi>
              x
             </mi>
<mo>
              ^
             </mo>
</mover>
<mo>
             +
            </mo>
<mtext>
             momentum
            </mtext>
<mo>
             ×
            </mo>
<msub>
<mi>
              x
             </mi>
<mi>
              t
             </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            hat{x}_text{new} = (1 - text{momentum}) times hat{x} + text{momentum} times x_t
           </annotation>
</semantics>
</math> -->x ^ new = ( 1 − momentum ) × x ^ + momentum × x t hat{x}_text{new} = (1 - text{momentum}) times hat{x} + text{momentum} times x_tx ^ new ​ = ( 1 − momentum ) × x ^ + momentum × x t ​  ,
where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mover accent="true">
<mi>
              x
             </mi>
<mo>
              ^
             </mo>
</mover>
</mrow>
<annotation encoding="application/x-tex">
            hat{x}
           </annotation>
</semantics>
</math> -->x ^ hat{x}x ^  is the estimated statistic and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              x
             </mi>
<mi>
              t
             </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            x_t
           </annotation>
</semantics>
</math> -->x t x_tx t ​  is the
new observed value.

Note 

[`InstanceNorm1d`](#torch.nn.InstanceNorm1d "torch.nn.InstanceNorm1d")  and [`LayerNorm`](torch.nn.LayerNorm.html#torch.nn.LayerNorm "torch.nn.LayerNorm")  are very similar, but
have some subtle differences. [`InstanceNorm1d`](#torch.nn.InstanceNorm1d "torch.nn.InstanceNorm1d")  is applied
on each channel of channeled data like multidimensional time series, but [`LayerNorm`](torch.nn.LayerNorm.html#torch.nn.LayerNorm "torch.nn.LayerNorm")  is usually applied on entire sample and often in NLP
tasks. Additionally, [`LayerNorm`](torch.nn.LayerNorm.html#torch.nn.LayerNorm "torch.nn.LayerNorm")  applies elementwise affine
transform, while [`InstanceNorm1d`](#torch.nn.InstanceNorm1d "torch.nn.InstanceNorm1d")  usually don’t apply affine
transform.

Parameters
:   * **num_features** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – number of features or channels <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                C
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               C
              </annotation>
</semantics>
</math> -->C CC  of the input

* **eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – a value added to the denominator for numerical stability. Default: 1e-5
* **momentum** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *]*  ) – the value used for the running_mean and running_var computation. Default: 0.1
* **affine** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – a boolean value that when set to `True`  , this module has
learnable affine parameters, initialized the same way as done for batch normalization.
Default: `False`  .
* **track_running_stats** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – a boolean value that when set to `True`  , this
module tracks the running mean and variance, and when set to `False`  ,
this module does not track such statistics and always uses batch
statistics in both training and eval modes. Default: `False`

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
<mi>
                C
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                L
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, L)
              </annotation>
</semantics>
</math> -->( N , C , L ) (N, C, L)( N , C , L )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                C
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                L
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (C, L)
              </annotation>
</semantics>
</math> -->( C , L ) (C, L)( C , L )

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
<mi>
                C
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                L
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, L)
              </annotation>
</semantics>
</math> -->( N , C , L ) (N, C, L)( N , C , L )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                C
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                L
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (C, L)
              </annotation>
</semantics>
</math> -->( C , L ) (C, L)( C , L )  (same shape as input)

Examples: 

```
>>> # Without Learnable Parameters
>>> m = nn.InstanceNorm1d(100)
>>> # With Learnable Parameters
>>> m = nn.InstanceNorm1d(100, affine=True)
>>> input = torch.randn(20, 100, 40)
>>> output = m(input)

```

