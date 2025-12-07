ELU 
==========================================

*class* torch.nn. ELU ( *alpha = 1.0*  , *inplace = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L522) 
:   Applies the Exponential Linear Unit (ELU) function, element-wise. 

Method described in the paper: [Fast and Accurate Deep Network Learning by Exponential Linear
Units (ELUs)](https://arxiv.org/abs/1511.07289)  . 

ELU is defined as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            ELU
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
                  &gt;
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
                  ≤
                 </mo>
<mn>
                  0
                 </mn>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           text{ELU}(x) = begin{cases}
x, &amp; text{ if } x &gt; 0
alpha * (exp(x) - 1), &amp; text{ if } x leq 0
end{cases}
          </annotation>
</semantics>
</math> -->
ELU ( x ) = { x , if x > 0 α ∗ ( exp ⁡ ( x ) − 1 ) , if x ≤ 0 text{ELU}(x) = begin{cases}
x, & text{ if } x > 0
alpha * (exp(x) - 1), & text{ if } x leq 0
end{cases}

ELU ( x ) = { x , α ∗ ( exp ( x ) − 1 ) , ​ if x > 0 if x ≤ 0 ​

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
</math> -->α alphaα  value for the ELU formulation. Default: 1.0

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

![../_images/ELU.png](../_images/ELU.png)

Examples: 

```
>>> m = nn.ELU()
>>> input = torch.randn(2)
>>> output = m(input)

```

