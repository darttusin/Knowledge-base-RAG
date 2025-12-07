torch.linalg.matrix_exp 
===================================================================================

torch.linalg. matrix_exp ( *A* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the matrix exponential of a square matrix. 

Letting <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="double-struck">
            K
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           mathbb{K}
          </annotation>
</semantics>
</math> -->K mathbb{K}K  be <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="double-struck">
            R
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           mathbb{R}
          </annotation>
</semantics>
</math> -->R mathbb{R}R  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="double-struck">
            C
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           mathbb{C}
          </annotation>
</semantics>
</math> -->C mathbb{C}C  ,
this function computes the **matrix exponential** of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            A
           </mi>
<mo>
            ∈
           </mo>
<msup>
<mi mathvariant="double-struck">
             K
            </mi>
<mrow>
<mi>
              n
             </mi>
<mo>
              ×
             </mo>
<mi>
              n
             </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           A in mathbb{K}^{n times n}
          </annotation>
</semantics>
</math> -->A ∈ K n × n A in mathbb{K}^{n times n}A ∈ K n × n  , which is defined as 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mrow>
<mi mathvariant="normal">
             m
            </mi>
<mi mathvariant="normal">
             a
            </mi>
<mi mathvariant="normal">
             t
            </mi>
<mi mathvariant="normal">
             r
            </mi>
<mi mathvariant="normal">
             i
            </mi>
<mi mathvariant="normal">
             x
            </mi>
<mi mathvariant="normal">
             _
            </mi>
<mi mathvariant="normal">
             e
            </mi>
<mi mathvariant="normal">
             x
            </mi>
<mi mathvariant="normal">
             p
            </mi>
</mrow>
<mo stretchy="false">
            (
           </mo>
<mi>
            A
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<munderover>
<mo>
             ∑
            </mo>
<mrow>
<mi>
              k
             </mi>
<mo>
              =
             </mo>
<mn>
              0
             </mn>
</mrow>
<mi mathvariant="normal">
             ∞
            </mi>
</munderover>
<mfrac>
<mn>
             1
            </mn>
<mrow>
<mi>
              k
             </mi>
<mo stretchy="false">
              !
             </mo>
</mrow>
</mfrac>
<msup>
<mi>
             A
            </mi>
<mi>
             k
            </mi>
</msup>
<mo>
            ∈
           </mo>
<msup>
<mi mathvariant="double-struck">
             K
            </mi>
<mrow>
<mi>
              n
             </mi>
<mo>
              ×
             </mo>
<mi>
              n
             </mi>
</mrow>
</msup>
<mi mathvariant="normal">
            .
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           mathrm{matrix_exp}(A) = sum_{k=0}^infty frac{1}{k!}A^k in mathbb{K}^{n times n}.
          </annotation>
</semantics>
</math> -->
m a t r i x _ e x p ( A ) = ∑ k = 0 ∞ 1 k ! A k ∈ K n × n . mathrm{matrix_exp}(A) = sum_{k=0}^infty frac{1}{k!}A^k in mathbb{K}^{n times n}.

matrix_exp ( A ) = k = 0 ∑ ∞ ​ k ! 1 ​ A k ∈ K n × n .

If the matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            A
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           A
          </annotation>
</semantics>
</math> -->A AA  has eigenvalues <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             λ
            </mi>
<mi>
             i
            </mi>
</msub>
<mo>
            ∈
           </mo>
<mi mathvariant="double-struck">
            C
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           lambda_i in mathbb{C}
          </annotation>
</semantics>
</math> -->λ i ∈ C lambda_i in mathbb{C}λ i ​ ∈ C  ,
the matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mrow>
<mi mathvariant="normal">
             m
            </mi>
<mi mathvariant="normal">
             a
            </mi>
<mi mathvariant="normal">
             t
            </mi>
<mi mathvariant="normal">
             r
            </mi>
<mi mathvariant="normal">
             i
            </mi>
<mi mathvariant="normal">
             x
            </mi>
<mi mathvariant="normal">
             _
            </mi>
<mi mathvariant="normal">
             e
            </mi>
<mi mathvariant="normal">
             x
            </mi>
<mi mathvariant="normal">
             p
            </mi>
</mrow>
<mo stretchy="false">
            (
           </mo>
<mi>
            A
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           mathrm{matrix_exp}(A)
          </annotation>
</semantics>
</math> -->m a t r i x _ e x p ( A ) mathrm{matrix_exp}(A)matrix_exp ( A )  has eigenvalues <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
             e
            </mi>
<msub>
<mi>
              λ
             </mi>
<mi>
              i
             </mi>
</msub>
</msup>
<mo>
            ∈
           </mo>
<mi mathvariant="double-struck">
            C
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           e^{lambda_i} in mathbb{C}
          </annotation>
</semantics>
</math> -->e λ i ∈ C e^{lambda_i} in mathbb{C}e λ i ​ ∈ C  . 

Supports input of bfloat16, float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if `A`  is a batch of matrices then
the output has the same batch dimensions. 

Parameters
: **A** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, n, n)* where *** is zero or more batch dimensions.

Example: 

```
>>> A = torch.empty(2, 2, 2)
>>> A[0, :, :] = torch.eye(2, 2)
>>> A[1, :, :] = 2 * torch.eye(2, 2)
>>> A
tensor([[[1., 0.],
         [0., 1.]],

        [[2., 0.],
         [0., 2.]]])
>>> torch.linalg.matrix_exp(A)
tensor([[[2.7183, 0.0000],
         [0.0000, 2.7183]],

         [[7.3891, 0.0000],
          [0.0000, 7.3891]]])

>>> import math
>>> A = torch.tensor([[0, math.pi/3], [-math.pi/3, 0]]) # A is skew-symmetric
>>> torch.linalg.matrix_exp(A) # matrix_exp(A) = [[cos(pi/3), sin(pi/3)], [-sin(pi/3), cos(pi/3)]]
tensor([[ 0.5000,  0.8660],
        [-0.8660,  0.5000]])

```

