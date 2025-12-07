LogSoftmax 
========================================================

*class* torch.nn. LogSoftmax ( *dim = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L1715) 
:   Applies the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            log
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<mtext>
            Softmax
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
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           log(text{Softmax}(x))
          </annotation>
</semantics>
</math> -->log ⁡ ( Softmax ( x ) ) log(text{Softmax}(x))lo g ( Softmax ( x ))  function to an n-dimensional input Tensor. 

The LogSoftmax formulation can be simplified as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            LogSoftmax
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
<mi>
            log
           </mi>
<mo>
            ⁡
           </mo>
<mrow>
<mo fence="true">
             (
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
<mo fence="true">
             )
            </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           text{LogSoftmax}(x_{i}) = logleft(frac{exp(x_i) }{ sum_j exp(x_j)} right)
          </annotation>
</semantics>
</math> -->
LogSoftmax ( x i ) = log ⁡ ( exp ⁡ ( x i ) ∑ j exp ⁡ ( x j ) ) text{LogSoftmax}(x_{i}) = logleft(frac{exp(x_i) }{ sum_j exp(x_j)} right)

LogSoftmax ( x i ​ ) = lo g ( ∑ j ​ exp ( x j ​ ) exp ( x i ​ ) ​ )

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

Parameters
: **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – A dimension along which LogSoftmax will be computed.

Returns
:   a Tensor of the same dimension and shape as the input with
values in the range [-inf, 0)

Return type
:   None

Examples: 

```
>>> m = nn.LogSoftmax(dim=1)
>>> input = torch.randn(2, 3)
>>> output = m(input)

```

