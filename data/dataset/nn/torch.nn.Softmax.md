Softmax 
==================================================

*class* torch.nn. Softmax ( *dim = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L1626) 
:   Applies the Softmax function to an n-dimensional input Tensor. 

Rescales them so that the elements of the n-dimensional output Tensor
lie in the range [0,1] and sum to 1. 

Softmax is defined as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            Softmax
           </mtext>
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             x
            </mi>
<mi>
             i
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mfrac>
<mrow>
<mi>
              exp
             </mi>
<mo>
              ⁡
             </mo>
<mo stretchy="false">
              (
             </mo>
<msub>
<mi>
               x
              </mi>
<mi>
               i
              </mi>
</msub>
<mo stretchy="false">
              )
             </mo>
</mrow>
<mrow>
<munder>
<mo>
               ∑
              </mo>
<mi>
               j
              </mi>
</munder>
<mi>
              exp
             </mi>
<mo>
              ⁡
             </mo>
<mo stretchy="false">
              (
             </mo>
<msub>
<mi>
               x
              </mi>
<mi>
               j
              </mi>
</msub>
<mo stretchy="false">
              )
             </mo>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           text{Softmax}(x_{i}) = frac{exp(x_i)}{sum_j exp(x_j)}
          </annotation>
</semantics>
</math> -->
Softmax ( x i ) = exp ⁡ ( x i ) ∑ j exp ⁡ ( x j ) text{Softmax}(x_{i}) = frac{exp(x_i)}{sum_j exp(x_j)}

Softmax ( x i ​ ) = ∑ j ​ exp ( x j ​ ) exp ( x i ​ ) ​

When the input Tensor is a sparse tensor then the unspecified
values are treated as `-inf`  . 

Shape:
:   * Input: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (*)
              </annotation>
</semantics>
</math> -->( ∗ ) (*)( ∗ )  where *** means, any number of additional
dimensions

* Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (*)
              </annotation>
</semantics>
</math> -->( ∗ ) (*)( ∗ )  , same shape as the input

Returns
:   a Tensor of the same dimension and shape as the input with
values in the range [0, 1]

Parameters
: **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – A dimension along which Softmax will be computed (so every slice
along dim will sum to 1).

Return type
:   None

Note 

This module doesn’t work directly with NLLLoss,
which expects the Log to be computed between the Softmax and itself.
Use *LogSoftmax* instead (it’s faster and has better numerical properties).

Examples: 

```
>>> m = nn.Softmax(dim=1)
>>> input = torch.randn(2, 3)
>>> output = m(input)

```

