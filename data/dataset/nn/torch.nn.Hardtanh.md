Hardtanh 
====================================================

*class* torch.nn. Hardtanh ( *min_val = -1.0*  , *max_val = 1.0*  , *inplace = False*  , *min_value = None*  , *max_value = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L202) 
:   Applies the HardTanh function element-wise. 

HardTanh is defined as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            HardTanh
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
<mtext>
                 max_val
                </mtext>
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
                  max_val
                 </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mtext>
                 min_val
                </mtext>
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
<mtext>
                  min_val
                 </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
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
           text{HardTanh}(x) = begin{cases}
    text{max_val} &amp; text{ if } x &gt; text{ max_val } 
    text{min_val} &amp; text{ if } x &lt; text{ min_val } 
    x &amp; text{ otherwise } 
end{cases}
          </annotation>
</semantics>
</math> -->
HardTanh ( x ) = { max_val if x > max_val min_val if x < min_val x otherwise text{HardTanh}(x) = begin{cases}
 text{max_val} & text{ if } x > text{ max_val } 
 text{min_val} & text{ if } x < text{ min_val } 
 x & text{ otherwise } 
end{cases}

HardTanh ( x ) = ⎩ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMzE2ZW0iIHByZXNlcnZlYXNwZWN0cmF0aW89InhNaW5ZTWluIiBzdHlsZT0id2lkdGg6MC44ODg5ZW0iIHZpZXdib3g9IjAgMCA4ODguODkgMzE2IiB3aWR0aD0iMC44ODg5ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0zODQgMCBINTA0IFYzMTYgSDM4NHogTTM4NCAwIEg1MDQgVjMxNiBIMzg0eiI+CjwvcGF0aD4KPC9zdmc+)⎨ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMzE2ZW0iIHByZXNlcnZlYXNwZWN0cmF0aW89InhNaW5ZTWluIiBzdHlsZT0id2lkdGg6MC44ODg5ZW0iIHZpZXdib3g9IjAgMCA4ODguODkgMzE2IiB3aWR0aD0iMC44ODg5ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0zODQgMCBINTA0IFYzMTYgSDM4NHogTTM4NCAwIEg1MDQgVjMxNiBIMzg0eiI+CjwvcGF0aD4KPC9zdmc+)⎧ ​ max_val min_val x ​ if x > max_val if x < min_val otherwise ​

Parameters
:   * **min_val** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – minimum value of the linear region range. Default: -1
* **max_val** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – maximum value of the linear region range. Default: 1
* **inplace** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – can optionally do the operation in-place. Default: `False`

Keyword arguments `min_value`  and `max_value`  have been deprecated in favor of `min_val`  and `max_val`  . 

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

![../_images/Hardtanh.png](../_images/Hardtanh.png)

Examples: 

```
>>> m = nn.Hardtanh(-2, 2)
>>> input = torch.randn(2)
>>> output = m(input)

```

