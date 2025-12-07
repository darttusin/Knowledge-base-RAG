torch.kron 
========================================================

torch. kron ( *input*  , *other*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the Kronecker product, denoted by <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
            ⊗
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           otimes
          </annotation>
</semantics>
</math> -->⊗ otimes⊗  , of `input`  and `other`  . 

If `input`  is a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             a
            </mi>
<mn>
             0
            </mn>
</msub>
<mo>
            ×
           </mo>
<msub>
<mi>
             a
            </mi>
<mn>
             1
            </mn>
</msub>
<mo>
            ×
           </mo>
<mo>
            ⋯
           </mo>
<mo>
            ×
           </mo>
<msub>
<mi>
             a
            </mi>
<mi>
             n
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (a_0 times a_1 times dots times a_n)
          </annotation>
</semantics>
</math> -->( a 0 × a 1 × ⋯ × a n ) (a_0 times a_1 times dots times a_n)( a 0 ​ × a 1 ​ × ⋯ × a n ​ )  tensor and `other`  is a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             b
            </mi>
<mn>
             0
            </mn>
</msub>
<mo>
            ×
           </mo>
<msub>
<mi>
             b
            </mi>
<mn>
             1
            </mn>
</msub>
<mo>
            ×
           </mo>
<mo>
            ⋯
           </mo>
<mo>
            ×
           </mo>
<msub>
<mi>
             b
            </mi>
<mi>
             n
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (b_0 times b_1 times dots times b_n)
          </annotation>
</semantics>
</math> -->( b 0 × b 1 × ⋯ × b n ) (b_0 times b_1 times dots times b_n)( b 0 ​ × b 1 ​ × ⋯ × b n ​ )  tensor, the result will be a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             a
            </mi>
<mn>
             0
            </mn>
</msub>
<mo>
            ∗
           </mo>
<msub>
<mi>
             b
            </mi>
<mn>
             0
            </mn>
</msub>
<mo>
            ×
           </mo>
<msub>
<mi>
             a
            </mi>
<mn>
             1
            </mn>
</msub>
<mo>
            ∗
           </mo>
<msub>
<mi>
             b
            </mi>
<mn>
             1
            </mn>
</msub>
<mo>
            ×
           </mo>
<mo>
            ⋯
           </mo>
<mo>
            ×
           </mo>
<msub>
<mi>
             a
            </mi>
<mi>
             n
            </mi>
</msub>
<mo>
            ∗
           </mo>
<msub>
<mi>
             b
            </mi>
<mi>
             n
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (a_0*b_0 times a_1*b_1 times dots times a_n*b_n)
          </annotation>
</semantics>
</math> -->( a 0 ∗ b 0 × a 1 ∗ b 1 × ⋯ × a n ∗ b n ) (a_0*b_0 times a_1*b_1 times dots times a_n*b_n)( a 0 ​ ∗ b 0 ​ × a 1 ​ ∗ b 1 ​ × ⋯ × a n ​ ∗ b n ​ )  tensor with the following entries: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mtext>
            input
           </mtext>
<mo>
            ⊗
           </mo>
<mtext>
            other
           </mtext>
<msub>
<mo stretchy="false">
             )
            </mo>
<mrow>
<msub>
<mi>
               k
              </mi>
<mn>
               0
              </mn>
</msub>
<mo separator="true">
              ,
             </mo>
<msub>
<mi>
               k
              </mi>
<mn>
               1
              </mn>
</msub>
<mo separator="true">
              ,
             </mo>
<mo>
              …
             </mo>
<mo separator="true">
              ,
             </mo>
<msub>
<mi>
               k
              </mi>
<mi>
               n
              </mi>
</msub>
</mrow>
</msub>
<mo>
            =
           </mo>
<msub>
<mtext>
             input
            </mtext>
<mrow>
<msub>
<mi>
               i
              </mi>
<mn>
               0
              </mn>
</msub>
<mo separator="true">
              ,
             </mo>
<msub>
<mi>
               i
              </mi>
<mn>
               1
              </mn>
</msub>
<mo separator="true">
              ,
             </mo>
<mo>
              …
             </mo>
<mo separator="true">
              ,
             </mo>
<msub>
<mi>
               i
              </mi>
<mi>
               n
              </mi>
</msub>
</mrow>
</msub>
<mo>
            ∗
           </mo>
<msub>
<mtext>
             other
            </mtext>
<mrow>
<msub>
<mi>
               j
              </mi>
<mn>
               0
              </mn>
</msub>
<mo separator="true">
              ,
             </mo>
<msub>
<mi>
               j
              </mi>
<mn>
               1
              </mn>
</msub>
<mo separator="true">
              ,
             </mo>
<mo>
              …
             </mo>
<mo separator="true">
              ,
             </mo>
<msub>
<mi>
               j
              </mi>
<mi>
               n
              </mi>
</msub>
</mrow>
</msub>
<mo separator="true">
            ,
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (text{input} otimes text{other})_{k_0, k_1, dots, k_n} =
    text{input}_{i_0, i_1, dots, i_n} * text{other}_{j_0, j_1, dots, j_n},
          </annotation>
</semantics>
</math> -->
( input ⊗ other ) k 0 , k 1 , … , k n = input i 0 , i 1 , … , i n ∗ other j 0 , j 1 , … , j n , (text{input} otimes text{other})_{k_0, k_1, dots, k_n} =
 text{input}_{i_0, i_1, dots, i_n} * text{other}_{j_0, j_1, dots, j_n},

( input ⊗ other ) k 0 ​ , k 1 ​ , … , k n ​ ​ = input i 0 ​ , i 1 ​ , … , i n ​ ​ ∗ other j 0 ​ , j 1 ​ , … , j n ​ ​ ,

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             k
            </mi>
<mi>
             t
            </mi>
</msub>
<mo>
            =
           </mo>
<msub>
<mi>
             i
            </mi>
<mi>
             t
            </mi>
</msub>
<mo>
            ∗
           </mo>
<msub>
<mi>
             b
            </mi>
<mi>
             t
            </mi>
</msub>
<mo>
            +
           </mo>
<msub>
<mi>
             j
            </mi>
<mi>
             t
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           k_t = i_t * b_t + j_t
          </annotation>
</semantics>
</math> -->k t = i t ∗ b t + j t k_t = i_t * b_t + j_tk t ​ = i t ​ ∗ b t ​ + j t ​  for <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
            0
           </mn>
<mo>
            ≤
           </mo>
<mi>
            t
           </mi>
<mo>
            ≤
           </mo>
<mi>
            n
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           0 leq t leq n
          </annotation>
</semantics>
</math> -->0 ≤ t ≤ n 0 leq t leq n0 ≤ t ≤ n  .
If one tensor has fewer dimensions than the other it is unsqueezed until it has the same number of dimensions. 

Supports real-valued and complex-valued inputs. 

Note 

This function generalizes the typical definition of the Kronecker product for two matrices to two tensors,
as described above. When `input`  is a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
             (
            </mo>
<mi>
             m
            </mi>
<mo>
             ×
            </mo>
<mi>
             n
            </mi>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            (m times n)
           </annotation>
</semantics>
</math> -->( m × n ) (m times n)( m × n )  matrix and `other`  is a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
             (
            </mo>
<mi>
             p
            </mi>
<mo>
             ×
            </mo>
<mi>
             q
            </mi>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            (p times q)
           </annotation>
</semantics>
</math> -->( p × q ) (p times q)( p × q )  matrix, the result will be a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
             (
            </mo>
<mi>
             p
            </mi>
<mo>
             ∗
            </mo>
<mi>
             m
            </mi>
<mo>
             ×
            </mo>
<mi>
             q
            </mi>
<mo>
             ∗
            </mo>
<mi>
             n
            </mi>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            (p*m times q*n)
           </annotation>
</semantics>
</math> -->( p ∗ m × q ∗ n ) (p*m times q*n)( p ∗ m × q ∗ n )  block matrix: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="bold">
             A
            </mi>
<mo>
             ⊗
            </mo>
<mi mathvariant="bold">
             B
            </mi>
<mo>
             =
            </mo>
<mrow>
<mo fence="true">
              [
             </mo>
<mtable columnalign="center center center" columnspacing="1em" rowspacing="0.16em">
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<msub>
<mi>
                    a
                   </mi>
<mn>
                    11
                   </mn>
</msub>
<mi mathvariant="bold">
                   B
                  </mi>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mo lspace="0em" rspace="0em">
                  ⋯
                 </mo>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<msub>
<mi>
                    a
                   </mi>
<mrow>
<mn>
                     1
                    </mn>
<mi>
                     n
                    </mi>
</mrow>
</msub>
<mi mathvariant="bold">
                   B
                  </mi>
</mrow>
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
<mrow>
<msub>
<mi>
                    a
                   </mi>
<mrow>
<mi>
                     m
                    </mi>
<mn>
                     1
                    </mn>
</mrow>
</msub>
<mi mathvariant="bold">
                   B
                  </mi>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mo lspace="0em" rspace="0em">
                  ⋯
                 </mo>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<msub>
<mi>
                    a
                   </mi>
<mrow>
<mi>
                     m
                    </mi>
<mi>
                     n
                    </mi>
</mrow>
</msub>
<mi mathvariant="bold">
                   B
                  </mi>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<mo fence="true">
              ]
             </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
            mathbf{A} otimes mathbf{B}=begin{bmatrix}
a_{11} mathbf{B} &amp; cdots &amp; a_{1 n} mathbf{B} 
vdots &amp; ddots &amp; vdots 
a_{m 1} mathbf{B} &amp; cdots &amp; a_{m n} mathbf{B} end{bmatrix}
           </annotation>
</semantics>
</math> -->
A ⊗ B = [ a 11 B ⋯ a 1 n B ⋮ ⋱ ⋮ a m 1 B ⋯ a m n B ] mathbf{A} otimes mathbf{B}=begin{bmatrix}
a_{11} mathbf{B} & cdots & a_{1 n} mathbf{B} 
vdots & ddots & vdots 
a_{m 1} mathbf{B} & cdots & a_{m n} mathbf{B} end{bmatrix}

A ⊗ B = ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjQuMjAwZW0iIHZpZXdib3g9IjAgMCA2NjcgNDIwMCIgd2lkdGg9IjAuNjY3ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik00MDMgMTc1OSBWODQgSDY2NiBWMCBIMzE5IFYxNzU5IHY2MDAgdjE3NTkgaDM0NyB2LTg0Ckg0MDN6IE00MDMgMTc1OSBWMCBIMzE5IFYxNzU5IHY2MDAgdjE3NTkgaDg0eiI+CjwvcGF0aD4KPC9zdmc+)​ a 11 ​ B ⋮ a m 1 ​ B ​ ⋯ ⋱ ⋯ ​ a 1 n ​ B ⋮ a mn ​ B ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjQuMjAwZW0iIHZpZXdib3g9IjAgMCA2NjcgNDIwMCIgd2lkdGg9IjAuNjY3ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0zNDcgMTc1OSBWMCBIMCBWODQgSDI2MyBWMTc1OSB2NjAwIHYxNzU5IEgwIHY4NCBIMzQ3egpNMzQ3IDE3NTkgVjAgSDI2MyBWMTc1OSB2NjAwIHYxNzU5IGg4NHoiPgo8L3BhdGg+Cjwvc3ZnPg==)​

where `input`  is <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="bold">
             A
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            mathbf{A}
           </annotation>
</semantics>
</math> -->A mathbf{A}A  and `other`  is <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="bold">
             B
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            mathbf{B}
           </annotation>
</semantics>
</math> -->B mathbf{B}B  .

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) –
* **other** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) –

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – The output tensor. Ignored if `None`  . Default: `None`

Examples: 

```
>>> mat1 = torch.eye(2)
>>> mat2 = torch.ones(2, 2)
>>> torch.kron(mat1, mat2)
tensor([[1., 1., 0., 0.],
        [1., 1., 0., 0.],
        [0., 0., 1., 1.],
        [0., 0., 1., 1.]])

>>> mat1 = torch.eye(2)
>>> mat2 = torch.arange(1, 5).reshape(2, 2)
>>> torch.kron(mat1, mat2)
tensor([[1., 2., 0., 0.],
        [3., 4., 0., 0.],
        [0., 0., 1., 2.],
        [0., 0., 3., 4.]])

```

