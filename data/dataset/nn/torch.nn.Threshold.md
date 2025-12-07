Threshold 
======================================================

*class* torch.nn. Threshold ( *threshold*  , *value*  , *inplace = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L48) 
:   Thresholds each element of the input Tensor. 

Threshold is defined as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            y
           </mi>
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
<mtext>
                  threshold
                 </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mtext>
                  value
                 </mtext>
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
           y =
begin{cases}
x, &amp;text{ if } x &gt; text{threshold} 
text{value}, &amp;text{ otherwise }
end{cases}
          </annotation>
</semantics>
</math> -->
y = { x , if x > threshold value , otherwise y =
begin{cases}
x, &text{ if } x > text{threshold} 
text{value}, &text{ otherwise }
end{cases}

y = { x , value , ​ if x > threshold otherwise ​

Parameters
:   * **threshold** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – The value to threshold at
* **value** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – The value to replace with
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

![../_images/Threshold.png](../_images/Threshold.png)

Examples: 

```
>>> m = nn.Threshold(0, 0.5)
>>> input = torch.arange(-3, 3)
>>> output = m(input)

```

