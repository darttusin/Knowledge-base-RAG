CELU 
============================================

*class* torch.nn. CELU ( *alpha = 1.0*  , *inplace = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L570) 
:   Applies the CELU function element-wise. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            CELU
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
<mi>
            max
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<mn>
            0
           </mn>
<mo separator="true">
            ,
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            +
           </mo>
<mi>
            min
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<mn>
            0
           </mn>
<mo separator="true">
            ,
           </mo>
<mi>
            α
           </mi>
<mo>
            ∗
           </mo>
<mo stretchy="false">
            (
           </mo>
<mi>
            exp
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mi mathvariant="normal">
            /
           </mi>
<mi>
            α
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            −
           </mo>
<mn>
            1
           </mn>
<mo stretchy="false">
            )
           </mo>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{CELU}(x) = max(0,x) + min(0, alpha * (exp(x/alpha) - 1))
          </annotation>
</semantics>
</math> -->
CELU ( x ) = max ⁡ ( 0 , x ) + min ⁡ ( 0 , α ∗ ( exp ⁡ ( x / α ) − 1 ) ) text{CELU}(x) = max(0,x) + min(0, alpha * (exp(x/alpha) - 1))

CELU ( x ) = max ( 0 , x ) + min ( 0 , α ∗ ( exp ( x / α ) − 1 ))

More details can be found in the paper [Continuously Differentiable Exponential Linear Units](https://arxiv.org/abs/1704.07483)  . 

Parameters
:   * **alpha** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                α
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               alpha
              </annotation>
</semantics>
</math> -->α alphaα  value for the CELU formulation. Default: 1.0

* **inplace** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – can optionally do the operation in-place. Default: `False`

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
</math> -->( ∗ ) (*)( ∗ )  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
                ∗
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               *
              </annotation>
</semantics>
</math> -->∗ *∗  means any number of dimensions.

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
</math> -->( ∗ ) (*)( ∗ )  , same shape as the input.

![../_images/CELU.png](../_images/CELU.png)

Examples: 

```
>>> m = nn.CELU()
>>> input = torch.randn(2)
>>> output = m(input)

```

