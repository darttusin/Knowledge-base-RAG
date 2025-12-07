GroupNorm 
======================================================

*class* torch.nn. GroupNorm ( *num_groups*  , *num_channels*  , *eps = 1e-05*  , *affine = True*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/normalization.py#L228) 
:   Applies Group Normalization over a mini-batch of inputs. 

This layer implements the operation as described in
the paper [Group Normalization](https://arxiv.org/abs/1803.08494) 

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

The input channels are separated into `num_groups`  groups, each containing `num_channels / num_groups`  channels. `num_channels`  must be divisible by `num_groups`  . The mean and standard-deviation are calculated
separately over the each group. <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->β betaβ  are learnable
per-channel affine transform parameter vectors of size `num_channels`  if `affine`  is `True`  .
The variance is calculated via the biased estimator, equivalent to *torch.var(input, unbiased=False)* . 

This layer uses statistics computed from input data in both training and
evaluation modes. 

Parameters
:   * **num_groups** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – number of groups to separate the channels into
* **num_channels** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – number of channels expected in input
* **eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – a value added to the denominator for numerical stability. Default: 1e-5
* **affine** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – a boolean value that when set to `True`  , this module
has learnable per-channel affine parameters initialized to ones (for weights)
and zeros (for biases). Default: `True`  .

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
<mo>
                ∗
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, *)
              </annotation>
</semantics>
</math> -->( N , C , ∗ ) (N, C, *)( N , C , ∗ )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                C
               </mi>
<mo>
                =
               </mo>
<mtext>
                num_channels
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               C=text{num_channels}
              </annotation>
</semantics>
</math> -->C = num_channels C=text{num_channels}C = num_channels

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
<mo>
                ∗
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, *)
              </annotation>
</semantics>
</math> -->( N , C , ∗ ) (N, C, *)( N , C , ∗ )  (same shape as input)

Examples: 

```
>>> input = torch.randn(20, 6, 10, 10)
>>> # Separate 6 channels into 3 groups
>>> m = nn.GroupNorm(3, 6)
>>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
>>> m = nn.GroupNorm(6, 6)
>>> # Put all 6 channels into a single group (equivalent with LayerNorm)
>>> m = nn.GroupNorm(1, 6)
>>> # Activating the module
>>> output = m(input)

```

