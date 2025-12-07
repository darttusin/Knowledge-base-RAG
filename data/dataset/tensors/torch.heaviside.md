torch.heaviside 
==================================================================

torch. heaviside ( *input*  , *values*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the Heaviside step function for each element in `input`  .
The Heaviside step function is defined as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            heaviside
           </mtext>
<mo stretchy="false">
            (
           </mo>
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
<mo separator="true">
            ,
           </mo>
<mi>
            v
           </mi>
<mi>
            a
           </mi>
<mi>
            l
           </mi>
<mi>
            u
           </mi>
<mi>
            e
           </mi>
<mi>
            s
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
                 if input &lt; 0
                </mtext>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mi>
                  v
                 </mi>
<mi>
                  a
                 </mi>
<mi>
                  l
                 </mi>
<mi>
                  u
                 </mi>
<mi>
                  e
                 </mi>
<mi>
                  s
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
                 if input == 0
                </mtext>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mn>
                  1
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
                 if input &gt; 0
                </mtext>
</mstyle>
</mtd>
</mtr>
</mtable>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           text{{heaviside}}(input, values) = begin{cases}
    0, &amp; text{if input &lt; 0}
    values, &amp; text{if input == 0}
    1, &amp; text{if input &gt; 0}
end{cases}
          </annotation>
</semantics>
</math> -->
heaviside ( i n p u t , v a l u e s ) = { 0 , if input < 0 v a l u e s , if input == 0 1 , if input > 0 text{{heaviside}}(input, values) = begin{cases}
 0, & text{if input < 0}
 values, & text{if input == 0}
 1, & text{if input > 0}
end{cases}

heaviside ( in p u t , v a l u es ) = ⎩ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMzE2ZW0iIHByZXNlcnZlYXNwZWN0cmF0aW89InhNaW5ZTWluIiBzdHlsZT0id2lkdGg6MC44ODg5ZW0iIHZpZXdib3g9IjAgMCA4ODguODkgMzE2IiB3aWR0aD0iMC44ODg5ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0zODQgMCBINTA0IFYzMTYgSDM4NHogTTM4NCAwIEg1MDQgVjMxNiBIMzg0eiI+CjwvcGF0aD4KPC9zdmc+)⎨ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMzE2ZW0iIHByZXNlcnZlYXNwZWN0cmF0aW89InhNaW5ZTWluIiBzdHlsZT0id2lkdGg6MC44ODg5ZW0iIHZpZXdib3g9IjAgMCA4ODguODkgMzE2IiB3aWR0aD0iMC44ODg5ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0zODQgMCBINTA0IFYzMTYgSDM4NHogTTM4NCAwIEg1MDQgVjMxNiBIMzg0eiI+CjwvcGF0aD4KPC9zdmc+)⎧ ​ 0 , v a l u es , 1 , ​ if input < 0 if input == 0 if input > 0 ​

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **values** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – The values to use where `input`  is zero.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> input = torch.tensor([-1.5, 0, 2.0])
>>> values = torch.tensor([0.5])
>>> torch.heaviside(input, values)
tensor([0.0000, 0.5000, 1.0000])
>>> values = torch.tensor([1.2, -2.0, 3.5])
>>> torch.heaviside(input, values)
tensor([0., -2., 1.])

```

