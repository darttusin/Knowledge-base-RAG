Softsign 
====================================================

*class* torch.nn. Softsign ( ** args*  , *** kwargs* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L1530) 
:   Applies the element-wise Softsign function. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            SoftSign
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
<mfrac>
<mi>
             x
            </mi>
<mrow>
<mn>
              1
             </mn>
<mo>
              +
             </mo>
<mi mathvariant="normal">
              ∣
             </mi>
<mi>
              x
             </mi>
<mi mathvariant="normal">
              ∣
             </mi>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           text{SoftSign}(x) = frac{x}{ 1 + |x|}
          </annotation>
</semantics>
</math> -->
SoftSign ( x ) = x 1 + ∣ x ∣ text{SoftSign}(x) = frac{x}{ 1 + |x|}

SoftSign ( x ) = 1 + ∣ x ∣ x ​

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

![../_images/Softsign.png](../_images/Softsign.png)

Examples: 

```
>>> m = nn.Softsign()
>>> input = torch.randn(2)
>>> output = m(input)

```

