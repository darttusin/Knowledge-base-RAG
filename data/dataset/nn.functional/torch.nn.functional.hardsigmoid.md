torch.nn.functional.hardsigmoid 
==================================================================================================

torch.nn.functional. hardsigmoid ( *input*  , *inplace = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L2284) 
:   Apply the Hardsigmoid function element-wise. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            Hardsigmoid
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
<mn>
                 0
                </mn>
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
<mo>
                  −
                 </mo>
<mn>
                  3
                 </mn>
<mo separator="true">
                  ,
                 </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mn>
                 1
                </mn>
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
<mo>
                  +
                 </mo>
<mn>
                  3
                 </mn>
<mo separator="true">
                  ,
                 </mo>
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
<mi mathvariant="normal">
                  /
                 </mi>
<mn>
                  6
                 </mn>
<mo>
                  +
                 </mo>
<mn>
                  1
                 </mn>
<mi mathvariant="normal">
                  /
                 </mi>
<mn>
                  2
                 </mn>
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
           text{Hardsigmoid}(x) = begin{cases}
    0 &amp; text{if~} x le -3, 
    1 &amp; text{if~} x ge +3, 
    x / 6 + 1 / 2 &amp; text{otherwise}
end{cases}
          </annotation>
</semantics>
</math> -->
Hardsigmoid ( x ) = { 0 if x ≤ − 3 , 1 if x ≥ + 3 , x / 6 + 1 / 2 otherwise text{Hardsigmoid}(x) = begin{cases}
 0 & text{if~} x le -3, 
 1 & text{if~} x ge +3, 
 x / 6 + 1 / 2 & text{otherwise}
end{cases}

Hardsigmoid ( x ) = ⎩ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMzE2ZW0iIHByZXNlcnZlYXNwZWN0cmF0aW89InhNaW5ZTWluIiBzdHlsZT0id2lkdGg6MC44ODg5ZW0iIHZpZXdib3g9IjAgMCA4ODguODkgMzE2IiB3aWR0aD0iMC44ODg5ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0zODQgMCBINTA0IFYzMTYgSDM4NHogTTM4NCAwIEg1MDQgVjMxNiBIMzg0eiI+CjwvcGF0aD4KPC9zdmc+)⎨ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjAuMzE2ZW0iIHByZXNlcnZlYXNwZWN0cmF0aW89InhNaW5ZTWluIiBzdHlsZT0id2lkdGg6MC44ODg5ZW0iIHZpZXdib3g9IjAgMCA4ODguODkgMzE2IiB3aWR0aD0iMC44ODg5ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0zODQgMCBINTA0IFYzMTYgSDM4NHogTTM4NCAwIEg1MDQgVjMxNiBIMzg0eiI+CjwvcGF0aD4KPC9zdmc+)⎧ ​ 0 1 x /6 + 1/2 ​ if x ≤ − 3 , if x ≥ + 3 , otherwise ​

Parameters
: **inplace** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If set to `True`  , will do this operation in-place. Default: `False`

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

See [`Hardsigmoid`](torch.nn.Hardsigmoid.html#torch.nn.Hardsigmoid "torch.nn.Hardsigmoid")  for more details.

