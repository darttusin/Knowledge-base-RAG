SiLU 
============================================

*class* torch.nn. SiLU ( *inplace = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L397) 
:   Applies the Sigmoid Linear Unit (SiLU) function, element-wise. 

The SiLU function is also known as the swish function. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            silu
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
            x
           </mi>
<mo>
            ∗
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
<mo separator="true">
            ,
           </mo>
<mtext>
            where
           </mtext>
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
<mtext>
            is the logistic sigmoid.
           </mtext>
</mrow>
<annotation encoding="application/x-tex">
           text{silu}(x) = x * sigma(x), text{where } sigma(x) text{ is the logistic sigmoid.}
          </annotation>
</semantics>
</math> -->
silu ( x ) = x ∗ σ ( x ) , where σ ( x ) is the logistic sigmoid. text{silu}(x) = x * sigma(x), text{where } sigma(x) text{ is the logistic sigmoid.}

silu ( x ) = x ∗ σ ( x ) , where σ ( x ) is the logistic sigmoid.

Note 

See [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)  where the SiLU (Sigmoid Linear Unit) was originally coined, and see [Sigmoid-Weighted Linear Units for Neural Network Function Approximation
in Reinforcement Learning](https://arxiv.org/abs/1702.03118)  and [Swish:
a Self-Gated Activation Function](https://arxiv.org/abs/1710.05941v1)  where the SiLU was experimented with later.

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

![../_images/SiLU.png](../_images/SiLU.png)

Examples: 

```
>>> m = nn.SiLU()
>>> input = torch.randn(2)
>>> output = m(input)

```

