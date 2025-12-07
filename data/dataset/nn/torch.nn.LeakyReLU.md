LeakyReLU 
======================================================

*class* torch.nn. LeakyReLU ( *negative_slope = 0.01*  , *inplace = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L787) 
:   Applies the LeakyReLU function element-wise. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            LeakyReLU
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
<mtext>
            negative_slope
           </mtext>
<mo>
            ∗
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
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{LeakyReLU}(x) = max(0, x) + text{negative_slope} * min(0, x)
          </annotation>
</semantics>
</math> -->
LeakyReLU ( x ) = max ⁡ ( 0 , x ) + negative_slope ∗ min ⁡ ( 0 , x ) text{LeakyReLU}(x) = max(0, x) + text{negative_slope} * min(0, x)

LeakyReLU ( x ) = max ( 0 , x ) + negative_slope ∗ min ( 0 , x )

or 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            LeakyReLU
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
<mrow>
<mo fence="true">
             {
            </mo>
<mtable columnalign="left left" columnspacing="1em" rowspacing="0.36em">
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mi>
                  x
                 </mi>
<mo separator="true">
                  ,
                 </mo>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mtext>
                  if
                 </mtext>
<mi>
                  x
                 </mi>
<mo>
                  ≥
                 </mo>
<mn>
                  0
                 </mn>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mtext>
                  negative_slope
                 </mtext>
<mo>
                  ×
                 </mo>
<mi>
                  x
                 </mi>
<mo separator="true">
                  ,
                 </mo>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mtext>
                 otherwise
                </mtext>
</mstyle>
</mtd>
</mtr>
</mtable>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           text{LeakyReLU}(x) =
begin{cases}
x, &amp; text{ if } x geq 0 
text{negative_slope} times x, &amp; text{ otherwise }
end{cases}
          </annotation>
</semantics>
</math> -->
LeakyReLU ( x ) = { x , if x ≥ 0 negative_slope × x , otherwise text{LeakyReLU}(x) =
begin{cases}
x, & text{ if } x geq 0 
text{negative_slope} times x, & text{ otherwise }
end{cases}

LeakyReLU ( x ) = { x , negative_slope × x , ​ if x ≥ 0 otherwise ​

Parameters
:   * **negative_slope** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – Controls the angle of the negative slope (which is used for
negative input values). Default: 1e-2
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

![../_images/LeakyReLU.png](../_images/LeakyReLU.png)

Examples: 

```
>>> m = nn.LeakyReLU(0.1)
>>> input = torch.randn(2)
>>> output = m(input)

```

