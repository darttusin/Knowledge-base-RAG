RMSNorm 
==================================================

*class* torch.nn. RMSNorm ( *normalized_shape*  , *eps = None*  , *elementwise_affine = True*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/normalization.py#L321) 
:   Applies Root Mean Square Layer Normalization over a mini-batch of inputs. 

This layer implements the operation as described in
the paper [Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467.pdf) 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             y
            </mi>
<mi>
             i
            </mi>
</msub>
<mo>
            =
           </mo>
<mfrac>
<msub>
<mi>
              x
             </mi>
<mi>
              i
             </mi>
</msub>
<mrow>
<mrow>
<mi mathvariant="normal">
               R
              </mi>
<mi mathvariant="normal">
               M
              </mi>
<mi mathvariant="normal">
               S
              </mi>
</mrow>
<mo stretchy="false">
              (
             </mo>
<mi>
              x
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
</mfrac>
<mo>
            ∗
           </mo>
<msub>
<mi>
             γ
            </mi>
<mi>
             i
            </mi>
</msub>
<mo separator="true">
            ,
           </mo>
<mspace width="1em">
</mspace>
<mtext>
            where
           </mtext>
<mspace width="1em">
</mspace>
<mtext>
            RMS
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<msqrt>
<mrow>
<mi>
              ϵ
             </mi>
<mo>
              +
             </mo>
<mfrac>
<mn>
               1
              </mn>
<mi>
               n
              </mi>
</mfrac>
<munderover>
<mo>
               ∑
              </mo>
<mrow>
<mi>
                i
               </mi>
<mo>
                =
               </mo>
<mn>
                1
               </mn>
</mrow>
<mi>
               n
              </mi>
</munderover>
<msubsup>
<mi>
               x
              </mi>
<mi>
               i
              </mi>
<mn>
               2
              </mn>
</msubsup>
</mrow>
</msqrt>
</mrow>
<annotation encoding="application/x-tex">
           y_i = frac{x_i}{mathrm{RMS}(x)} * gamma_i, quad
text{where} quad text{RMS}(x) = sqrt{epsilon + frac{1}{n} sum_{i=1}^{n} x_i^2}
          </annotation>
</semantics>
</math> -->
y i = x i R M S ( x ) ∗ γ i , where RMS ( x ) = ϵ + 1 n ∑ i = 1 n x i 2 y_i = frac{x_i}{mathrm{RMS}(x)} * gamma_i, quad
text{where} quad text{RMS}(x) = sqrt{epsilon + frac{1}{n} sum_{i=1}^{n} x_i^2}

y i ​ = RMS ( x ) x i ​ ​ ∗ γ i ​ , where RMS ( x ) = ϵ + n 1 ​ i = 1 ∑ n ​ x i 2 ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjMuMTk2OGVtIiBwcmVzZXJ2ZWFzcGVjdHJhdGlvPSJ4TWluWU1pbiBzbGljZSIgdmlld2JveD0iMCAwIDQwMDAwMCAzMTk2IiB3aWR0aD0iNDAwZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik03MDIgODBINDAwMDAwNDAKSDc0MnYzMDYybC00IDQtNCA0Yy0uNjY3LjcgLTIgMS41LTQgMi41cy00LjE2NyAxLjgzMy02LjUgMi41LTUuNSAxLTkuNSAxCmgtMTJsLTI4LTg0Yy0xNi42NjctNTItOTYuNjY3IC0yOTQuMzMzLTI0MC03MjdsLTIxMiAtNjQzIC04NSAxNzAKYy00LTMuMzMzLTguMzMzLTcuNjY3LTEzIC0xM2wtMTMtMTNsNzctMTU1IDc3LTE1NmM2NiAxOTkuMzMzIDEzOSA0MTkuNjY3CjIxOSA2NjEgbDIxOCA2NjF6TTcwMiA4MEg0MDAwMDB2NDBINzQyeiI+CjwvcGF0aD4KPC9zdmc+)​

The RMS is taken over the last `D`  dimensions, where `D`  is the dimension of `normalized_shape`  . For example, if `normalized_shape`  is `(3, 5)`  (a 2-dimensional shape), the RMS is computed over
the last 2 dimensions of the input. 

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

* **eps** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *]*  ) – a value added to the denominator for numerical stability. Default: `torch.finfo(x.dtype).eps`
* **elementwise_affine** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – a boolean value that when set to `True`  , this module
has learnable per-element affine parameters initialized to ones (for weights). Default: `True`  .

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
>>> rms_norm = nn.RMSNorm([2, 3])
>>> input = torch.randn(2, 2, 3)
>>> rms_norm(input)

```

extra_repr ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/normalization.py#L404) 
:   Extra information about the module. 

Return type
:   [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")

forward ( *x* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/normalization.py#L398) 
:   Runs forward pass. 

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

reset_parameters ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/normalization.py#L391) 
:   Resets parameters based on their initialization used in __init__.

