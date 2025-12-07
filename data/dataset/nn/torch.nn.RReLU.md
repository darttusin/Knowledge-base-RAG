RReLU 
==============================================

*class* torch.nn. RReLU ( *lower = 0.125*  , *upper = 0.3333333333333333*  , *inplace = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L142) 
:   Applies the randomized leaky rectified linear unit function, element-wise. 

Method described in the paper: [Empirical Evaluation of Rectified Activations in Convolutional Network](https://arxiv.org/abs/1505.00853)  . 

The function is defined as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            RReLU
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
<mi>
                 x
                </mi>
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
<mi>
                  a
                 </mi>
<mi>
                  x
                 </mi>
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
           text{RReLU}(x) =
begin{cases}
    x &amp; text{if } x geq 0 
    ax &amp; text{ otherwise }
end{cases}
          </annotation>
</semantics>
</math> -->
RReLU ( x ) = { x if x ≥ 0 a x otherwise text{RReLU}(x) =
begin{cases}
 x & text{if } x geq 0 
 ax & text{ otherwise }
end{cases}

RReLU ( x ) = { x a x ​ if x ≥ 0 otherwise ​

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            a
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           a
          </annotation>
</semantics>
</math> -->a aa  is randomly sampled from uniform distribution <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="script">
            U
           </mi>
<mo stretchy="false">
            (
           </mo>
<mtext>
            lower
           </mtext>
<mo separator="true">
            ,
           </mo>
<mtext>
            upper
           </mtext>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           mathcal{U}(text{lower}, text{upper})
          </annotation>
</semantics>
</math> -->U ( lower , upper ) mathcal{U}(text{lower}, text{upper})U ( lower , upper )  during training while during
evaluation <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            a
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           a
          </annotation>
</semantics>
</math> -->a aa  is fixed with <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            a
           </mi>
<mo>
            =
           </mo>
<mfrac>
<mrow>
<mtext>
              lower
             </mtext>
<mo>
              +
             </mo>
<mtext>
              upper
             </mtext>
</mrow>
<mn>
             2
            </mn>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           a = frac{text{lower} + text{upper}}{2}
          </annotation>
</semantics>
</math> -->a = lower + upper 2 a = frac{text{lower} + text{upper}}{2}a = 2 lower + upper ​  . 

Parameters
:   * **lower** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – lower bound of the uniform distribution. Default: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mfrac>
<mn>
                 1
                </mn>
<mn>
                 8
                </mn>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
               frac{1}{8}
              </annotation>
</semantics>
</math> -->1 8 frac{1}{8}8 1 ​

* **upper** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – upper bound of the uniform distribution. Default: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mfrac>
<mn>
                 1
                </mn>
<mn>
                 3
                </mn>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
               frac{1}{3}
              </annotation>
</semantics>
</math> -->1 3 frac{1}{3}3 1 ​

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

![../_images/RReLU.png](../_images/RReLU.png)

Examples: 

```
>>> m = nn.RReLU(0.1, 0.3)
>>> input = torch.randn(2)
>>> output = m(input)

```

