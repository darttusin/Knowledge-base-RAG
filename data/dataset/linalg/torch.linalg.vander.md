torch.linalg.vander 
==========================================================================

torch.linalg. vander ( *x*  , *N = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Generates a Vandermonde matrix. 

Returns the Vandermonde matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            V
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           V
          </annotation>
</semantics>
</math> -->V VV 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            V
           </mi>
<mo>
            =
           </mo>
<mrow>
<mo fence="true">
             (
            </mo>
<mtable columnalign="center center center center center" columnspacing="1em" rowspacing="0.16em">
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
<msub>
<mi>
                  x
                 </mi>
<mn>
                  1
                 </mn>
</msub>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<msubsup>
<mi>
                  x
                 </mi>
<mn>
                  1
                 </mn>
<mn>
                  2
                 </mn>
</msubsup>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mo lspace="0em" rspace="0em">
                 …
                </mo>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<msubsup>
<mi>
                  x
                 </mi>
<mn>
                  1
                 </mn>
<mrow>
<mi>
                   N
                  </mi>
<mo>
                   −
                  </mo>
<mn>
                   1
                  </mn>
</mrow>
</msubsup>
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
<msub>
<mi>
                  x
                 </mi>
<mn>
                  2
                 </mn>
</msub>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<msubsup>
<mi>
                  x
                 </mi>
<mn>
                  2
                 </mn>
<mn>
                  2
                 </mn>
</msubsup>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mo lspace="0em" rspace="0em">
                 …
                </mo>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<msubsup>
<mi>
                  x
                 </mi>
<mn>
                  2
                 </mn>
<mrow>
<mi>
                   N
                  </mi>
<mo>
                   −
                  </mo>
<mn>
                   1
                  </mn>
</mrow>
</msubsup>
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
<msub>
<mi>
                  x
                 </mi>
<mn>
                  3
                 </mn>
</msub>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<msubsup>
<mi>
                  x
                 </mi>
<mn>
                  3
                 </mn>
<mn>
                  2
                 </mn>
</msubsup>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mo lspace="0em" rspace="0em">
                 …
                </mo>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<msubsup>
<mi>
                  x
                 </mi>
<mn>
                  3
                 </mn>
<mrow>
<mi>
                   N
                  </mi>
<mo>
                   −
                  </mo>
<mn>
                   1
                  </mn>
</mrow>
</msubsup>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mi mathvariant="normal">
                  ⋮
                 </mi>
<mpadded height="0em" voffset="0em">
<mspace height="1.5em" mathbackground="black" width="0em">
</mspace>
</mpadded>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mi mathvariant="normal">
                  ⋮
                 </mi>
<mpadded height="0em" voffset="0em">
<mspace height="1.5em" mathbackground="black" width="0em">
</mspace>
</mpadded>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mi mathvariant="normal">
                  ⋮
                 </mi>
<mpadded height="0em" voffset="0em">
<mspace height="1.5em" mathbackground="black" width="0em">
</mspace>
</mpadded>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mo lspace="0em" rspace="0em">
                 ⋱
                </mo>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mi mathvariant="normal">
                  ⋮
                 </mi>
<mpadded height="0em" voffset="0em">
<mspace height="1.5em" mathbackground="black" width="0em">
</mspace>
</mpadded>
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
<msub>
<mi>
                  x
                 </mi>
<mi>
                  n
                 </mi>
</msub>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<msubsup>
<mi>
                  x
                 </mi>
<mi>
                  n
                 </mi>
<mn>
                  2
                 </mn>
</msubsup>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mo lspace="0em" rspace="0em">
                 …
                </mo>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<msubsup>
<mi>
                  x
                 </mi>
<mi>
                  n
                 </mi>
<mrow>
<mi>
                   N
                  </mi>
<mo>
                   −
                  </mo>
<mn>
                   1
                  </mn>
</mrow>
</msubsup>
</mstyle>
</mtd>
</mtr>
</mtable>
<mo fence="true">
             )
            </mo>
</mrow>
<mi mathvariant="normal">
            .
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           V = begin{pmatrix}
        1 &amp; x_1 &amp; x_1^2 &amp; dots &amp; x_1^{N-1}
        1 &amp; x_2 &amp; x_2^2 &amp; dots &amp; x_2^{N-1}
        1 &amp; x_3 &amp; x_3^2 &amp; dots &amp; x_3^{N-1}
        vdots &amp; vdots &amp; vdots &amp; ddots &amp;vdots 
        1 &amp; x_n &amp; x_n^2 &amp; dots &amp; x_n^{N-1}
    end{pmatrix}.
          </annotation>
</semantics>
</math> -->
V = ( 1 x 1 x 1 2 … x 1 N − 1 1 x 2 x 2 2 … x 2 N − 1 1 x 3 x 3 2 … x 3 N − 1 ⋮ ⋮ ⋮ ⋱ ⋮ 1 x n x n 2 … x n N − 1 ) . V = begin{pmatrix}
 1 & x_1 & x_1^2 & dots & x_1^{N-1}
 1 & x_2 & x_2^2 & dots & x_2^{N-1}
 1 & x_3 & x_3^2 & dots & x_3^{N-1}
 vdots & vdots & vdots & ddots &vdots 
 1 & x_n & x_n^2 & dots & x_n^{N-1}
 end{pmatrix}.

V = ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjYuNjAwZW0iIHZpZXdib3g9IjAgMCA4NzUgNjYwMCIgd2lkdGg9IjAuODc1ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik04NjMsOWMwLC0yLC0yLC01LC02LC05YzAsMCwtMTcsMCwtMTcsMGMtMTIuNywwLC0xOS4zLDAuMywtMjAsMQpjLTUuMyw1LjMsLTEwLjMsMTEsLTE1LDE3Yy0yNDIuNywyOTQuNywtMzk1LjMsNjgyLC00NTgsMTE2MmMtMjEuMywxNjMuMywtMzMuMywzNDksCi0zNiw1NTcgbDAsMzA4NGMwLjIsNiwwLDI2LDAsNjBjMiwxNTkuMywxMCwzMTAuNywyNCw0NTRjNTMuMyw1MjgsMjEwLAo5NDkuNyw0NzAsMTI2NWM0LjcsNiw5LjcsMTEuNywxNSwxN2MwLjcsMC43LDcsMSwxOSwxYzAsMCwxOCwwLDE4LDBjNCwtNCw2LC03LDYsLTkKYzAsLTIuNywtMy4zLC04LjcsLTEwLC0xOGMtMTM1LjMsLTE5Mi43LC0yMzUuNSwtNDE0LjMsLTMwMC41LC02NjVjLTY1LC0yNTAuNywtMTAyLjUsCi01NDQuNywtMTEyLjUsLTg4MmMtMiwtMTA0LC0zLC0xNjcsLTMsLTE4OQpsMCwtMzA5MmMwLC0xNjIuNyw1LjcsLTMxNCwxNywtNDU0YzIwLjcsLTI3Miw2My43LC01MTMsMTI5LC03MjNjNjUuMywKLTIxMCwxNTUuMywtMzk2LjMsMjcwLC01NTljNi43LC05LjMsMTAsLTE1LjMsMTAsLTE4eiI+CjwvcGF0aD4KPC9zdmc+)​ 1 1 1 ⋮ 1 ​ x 1 ​ x 2 ​ x 3 ​ ⋮ x n ​ ​ x 1 2 ​ x 2 2 ​ x 3 2 ​ ⋮ x n 2 ​ ​ … … … ⋱ … ​ x 1 N − 1 ​ x 2 N − 1 ​ x 3 N − 1 ​ ⋮ x n N − 1 ​ ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjYuNjAwZW0iIHZpZXdib3g9IjAgMCA4NzUgNjYwMCIgd2lkdGg9IjAuODc1ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik03NiwwYy0xNi43LDAsLTI1LDMsLTI1LDljMCwyLDIsNi4zLDYsMTNjMjEuMywyOC43LDQyLjMsNjAuMywKNjMsOTVjOTYuNywxNTYuNywxNzIuOCwzMzIuNSwyMjguNSw1MjcuNWM1NS43LDE5NSw5Mi44LDQxNi41LDExMS41LDY2NC41CmMxMS4zLDEzOS4zLDE3LDI5MC43LDE3LDQ1NGMwLDI4LDEuNyw0MywzLjMsNDVsMCwzMDA5CmMtMyw0LC0zLjMsMTYuNywtMy4zLDM4YzAsMTYyLC01LjcsMzEzLjcsLTE3LDQ1NWMtMTguNywyNDgsLTU1LjgsNDY5LjMsLTExMS41LDY2NApjLTU1LjcsMTk0LjcsLTEzMS44LDM3MC4zLC0yMjguNSw1MjdjLTIwLjcsMzQuNywtNDEuNyw2Ni4zLC02Myw5NWMtMiwzLjMsLTQsNywtNiwxMQpjMCw3LjMsNS43LDExLDE3LDExYzAsMCwxMSwwLDExLDBjOS4zLDAsMTQuMywtMC4zLDE1LC0xYzUuMywtNS4zLDEwLjMsLTExLDE1LC0xNwpjMjQyLjcsLTI5NC43LDM5NS4zLC02ODEuNyw0NTgsLTExNjFjMjEuMywtMTY0LjcsMzMuMywtMzUwLjcsMzYsLTU1OApsMCwtMzE0NGMtMiwtMTU5LjMsLTEwLC0zMTAuNywtMjQsLTQ1NGMtNTMuMywtNTI4LC0yMTAsLTk0OS43LAotNDcwLC0xMjY1Yy00LjcsLTYsLTkuNywtMTEuNywtMTUsLTE3Yy0wLjcsLTAuNywtNi43LC0xLC0xOCwtMXoiPgo8L3BhdGg+Cjwvc3ZnPg==)​ .

for *N > 1* .
If `N` *= None* , then *N = x.size(-1)* so that the output is a square matrix. 

Supports inputs of float, double, cfloat, cdouble, and integral dtypes.
Also supports batches of vectors, and if `x`  is a batch of vectors then
the output has the same batch dimensions. 

Differences with *numpy.vander* : 

* Unlike *numpy.vander* , this function returns the powers of `x`  in ascending order.
To get them in the reverse order call `linalg.vander(x, N).flip(-1)`  .

Parameters
: **x** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, n)* where *** is zero or more batch dimensions
consisting of vectors.

Keyword Arguments
: **N** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – Number of columns in the output. Default: *x.size(-1)*

Example: 

```
>>> x = torch.tensor([1, 2, 3, 5])
>>> linalg.vander(x)
tensor([[  1,   1,   1,   1],
        [  1,   2,   4,   8],
        [  1,   3,   9,  27],
        [  1,   5,  25, 125]])
>>> linalg.vander(x, N=3)
tensor([[ 1,  1,  1],
        [ 1,  2,  4],
        [ 1,  3,  9],
        [ 1,  5, 25]])

```

