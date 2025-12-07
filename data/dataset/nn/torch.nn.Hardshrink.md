Hardshrink 
========================================================

*class* torch.nn. Hardshrink ( *lambd = 0.5* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L744) 
:   Applies the Hard Shrinkage (Hardshrink) function element-wise. 

Hardshrink is defined as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            HardShrink
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
<mi>
                  λ
                 </mi>
</mrow>
</mstyle>
</mtd>
</mtr>
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
                  &lt;
                 </mo>
<mo>
                  −
                 </mo>
<mi>
                  λ
                 </mi>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mn>
                  0
                 </mn>
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
           text{HardShrink}(x) =
begin{cases}
x, &amp; text{ if } x &gt; lambda 
x, &amp; text{ if } x &lt; -lambda 
0, &amp; text{ otherwise }
end{cases}
          </annotation>
</semantics>
</math> -->
HardShrink ( x ) = { x , if x > λ x , if x < − λ 0 , otherwise text{HardShrink}(x) =
begin{cases}
x, & text{ if } x > lambda 
x, & text{ if } x < -lambda 
0, & text{ otherwise }
end{cases}

HardShrink ( x ) = ⎩ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMzE2ZW0iIHByZXNlcnZlYXNwZWN0cmF0aW89InhNaW5ZTWluIiBzdHlsZT0id2lkdGg6MC44ODg5ZW0iIHZpZXdib3g9IjAgMCA4ODguODkgMzE2IiB3aWR0aD0iMC44ODg5ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0zODQgMCBINTA0IFYzMTYgSDM4NHogTTM4NCAwIEg1MDQgVjMxNiBIMzg0eiI+CjwvcGF0aD4KPC9zdmc+)⎨ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMzE2ZW0iIHByZXNlcnZlYXNwZWN0cmF0aW89InhNaW5ZTWluIiBzdHlsZT0id2lkdGg6MC44ODg5ZW0iIHZpZXdib3g9IjAgMCA4ODguODkgMzE2IiB3aWR0aD0iMC44ODg5ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0zODQgMCBINTA0IFYzMTYgSDM4NHogTTM4NCAwIEg1MDQgVjMxNiBIMzg0eiI+CjwvcGF0aD4KPC9zdmc+)⎧ ​ x , x , 0 , ​ if x > λ if x < − λ otherwise ​

Parameters
: **lambd** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
              λ
             </mi>
</mrow>
<annotation encoding="application/x-tex">
             lambda
            </annotation>
</semantics>
</math> -->λ lambdaλ  value for the Hardshrink formulation. Default: 0.5

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

![../_images/Hardshrink.png](../_images/Hardshrink.png)

Examples: 

```
>>> m = nn.Hardshrink()
>>> input = torch.randn(2)
>>> output = m(input)

```

