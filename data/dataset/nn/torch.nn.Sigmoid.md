Sigmoid 
==================================================

*class* torch.nn. Sigmoid ( ** args*  , *** kwargs* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L308) 
:   Applies the Sigmoid function element-wise. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            Sigmoid
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
            σ
           </mi>
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
<mn>
             1
            </mn>
<mrow>
<mn>
              1
             </mn>
<mo>
              +
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
<mo>
              −
             </mo>
<mi>
              x
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           text{Sigmoid}(x) = sigma(x) = frac{1}{1 + exp(-x)}
          </annotation>
</semantics>
</math> -->
Sigmoid ( x ) = σ ( x ) = 1 1 + exp ⁡ ( − x ) text{Sigmoid}(x) = sigma(x) = frac{1}{1 + exp(-x)}

Sigmoid ( x ) = σ ( x ) = 1 + exp ( − x ) 1 ​

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

![../_images/Sigmoid.png](../_images/Sigmoid.png)

Examples: 

```
>>> m = nn.Sigmoid()
>>> input = torch.randn(2)
>>> output = m(input)

```

