Softplus 
====================================================

*class* torch.nn. Softplus ( *beta = 1.0*  , *threshold = 20.0* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L862) 
:   Applies the Softplus function element-wise. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            Softplus
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
<mn>
             1
            </mn>
<mi>
             β
            </mi>
</mfrac>
<mo>
            ∗
           </mo>
<mi>
            log
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
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
<mi>
            β
           </mi>
<mo>
            ∗
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
           text{Softplus}(x) = frac{1}{beta} * log(1 + exp(beta * x))
          </annotation>
</semantics>
</math> -->
Softplus ( x ) = 1 β ∗ log ⁡ ( 1 + exp ⁡ ( β ∗ x ) ) text{Softplus}(x) = frac{1}{beta} * log(1 + exp(beta * x))

Softplus ( x ) = β 1 ​ ∗ lo g ( 1 + exp ( β ∗ x ))

SoftPlus is a smooth approximation to the ReLU function and can be used
to constrain the output of a machine to always be positive. 

For numerical stability the implementation reverts to the linear function
when <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            i
           </mi>
<mi>
            n
           </mi>
<mi>
            p
           </mi>
<mi>
            u
           </mi>
<mi>
            t
           </mi>
<mo>
            ×
           </mo>
<mi>
            β
           </mi>
<mo>
            &gt;
           </mo>
<mi>
            t
           </mi>
<mi>
            h
           </mi>
<mi>
            r
           </mi>
<mi>
            e
           </mi>
<mi>
            s
           </mi>
<mi>
            h
           </mi>
<mi>
            o
           </mi>
<mi>
            l
           </mi>
<mi>
            d
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           input times beta &gt; threshold
          </annotation>
</semantics>
</math> -->i n p u t × β > t h r e s h o l d input times beta > thresholdin p u t × β > t h res h o l d  . 

Parameters
:   * **beta** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->β betaβ  value for the Softplus formulation. Default: 1

* **threshold** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – values above this revert to a linear function. Default: 20

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

![../_images/Softplus.png](../_images/Softplus.png)

Examples: 

```
>>> m = nn.Softplus()
>>> input = torch.randn(2)
>>> output = m(input)

```

