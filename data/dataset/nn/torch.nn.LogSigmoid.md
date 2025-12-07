LogSigmoid 
========================================================

*class* torch.nn. LogSigmoid ( ** args*  , *** kwargs* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L839) 
:   Applies the Logsigmoid function element-wise. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            LogSigmoid
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
<mo fence="true">
             )
            </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           text{LogSigmoid}(x) = logleft(frac{ 1 }{ 1 + exp(-x)}right)
          </annotation>
</semantics>
</math> -->
LogSigmoid ( x ) = log ⁡ ( 1 1 + exp ⁡ ( − x ) ) text{LogSigmoid}(x) = logleft(frac{ 1 }{ 1 + exp(-x)}right)

LogSigmoid ( x ) = lo g ( 1 + exp ( − x ) 1 ​ )

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

![../_images/LogSigmoid.png](../_images/LogSigmoid.png)

Examples: 

```
>>> m = nn.LogSigmoid()
>>> input = torch.randn(2)
>>> output = m(input)

```

