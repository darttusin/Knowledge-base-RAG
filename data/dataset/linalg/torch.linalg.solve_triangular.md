torch.linalg.solve_triangular 
===============================================================================================

torch.linalg. solve_triangular ( *A*  , *B*  , *** , *upper*  , *left = True*  , *unitriangular = False*  , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the solution of a triangular system of linear equations with a unique solution. 

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
this function computes the solution <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            X
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
              k
             </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           X in mathbb{K}^{n times k}
          </annotation>
</semantics>
</math> -->X ∈ K n × k X in mathbb{K}^{n times k}X ∈ K n × k  of the **linear system** associated to the triangular matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A ∈ K n × n A in mathbb{K}^{n times n}A ∈ K n × n  without zeros on the diagonal
(that is, it is [invertible](https://en.wikipedia.org/wiki/Invertible_matrix#The_invertible_matrix_theorem)  ) and the rectangular matrix , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            B
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
              k
             </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           B in mathbb{K}^{n times k}
          </annotation>
</semantics>
</math> -->B ∈ K n × k B in mathbb{K}^{n times k}B ∈ K n × k  ,
which is defined as 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            A
           </mi>
<mi>
            X
           </mi>
<mo>
            =
           </mo>
<mi>
            B
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           AX = B
          </annotation>
</semantics>
</math> -->
A X = B AX = B

A X = B

The argument `upper`  signals whether <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  is upper or lower triangular. 

If `left` *= False* , this function returns the matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            X
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
              k
             </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           X in mathbb{K}^{n times k}
          </annotation>
</semantics>
</math> -->X ∈ K n × k X in mathbb{K}^{n times k}X ∈ K n × k  that
solves the system 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            X
           </mi>
<mi>
            A
           </mi>
<mo>
            =
           </mo>
<mi>
            B
           </mi>
<mpadded width="0px">
<mrow>
<mspace width="2em">
</mspace>
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
                k
               </mi>
<mo>
                ×
               </mo>
<mi>
                k
               </mi>
</mrow>
</msup>
<mo separator="true">
              ,
             </mo>
<mi>
              B
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
                k
               </mi>
</mrow>
</msup>
<mi mathvariant="normal">
              .
             </mi>
</mrow>
</mpadded>
</mrow>
<annotation encoding="application/x-tex">
           XA = Bmathrlap{qquad A in mathbb{K}^{k times k}, B in mathbb{K}^{n times k}.}
          </annotation>
</semantics>
</math> -->
X A = B A ∈ K k × k , B ∈ K n × k . XA = Bmathrlap{qquad A in mathbb{K}^{k times k}, B in mathbb{K}^{n times k}.}

X A = B A ∈ K k × k , B ∈ K n × k .

If `upper` *= True* (resp. *False* ) just the upper (resp. lower) triangular half of `A`  will be accessed. The elements below the main diagonal will be considered to be zero and will not be accessed. 

If `unitriangular` *= True* , the diagonal of `A`  is assumed to be ones and will not be accessed. 

The result may contain *NaN* s if the diagonal of `A`  contains zeros or elements that
are very close to zero and `unitriangular` *= False* (default) or if the input matrix
has very small eigenvalues. 

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if the inputs are batches of matrices then
the output has the same batch dimensions. 

See also 

[`torch.linalg.solve()`](torch.linalg.solve.html#torch.linalg.solve "torch.linalg.solve")  computes the solution of a general square system of linear
equations with a unique solution.

Parameters
:   * **A** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, n, n)* (or *(*, k, k)* if `left` *= False* )
where *** is zero or more batch dimensions.
* **B** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – right-hand side tensor of shape *(*, n, k)* .

Keyword Arguments
:   * **upper** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – whether `A`  is an upper or lower triangular matrix.
* **left** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether to solve the system <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                A
               </mi>
<mi>
                X
               </mi>
<mo>
                =
               </mo>
<mi>
                B
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               AX=B
              </annotation>
</semantics>
</math> -->A X = B AX=BA X = B  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                X
               </mi>
<mi>
                A
               </mi>
<mo>
                =
               </mo>
<mi>
                B
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               XA = B
              </annotation>
</semantics>
</math> -->X A = B XA = BX A = B  . Default: *True* .

* **unitriangular** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – if *True* , the diagonal elements of `A`  are assumed to be
all equal to *1* . Default: *False* .
* **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – output tensor. *B* may be passed as *out* and the result is computed in-place on *B* .
Ignored if *None* . Default: *None* .

Examples: 

```
>>> A = torch.randn(3, 3).triu_()
>>> B = torch.randn(3, 4)
>>> X = torch.linalg.solve_triangular(A, B, upper=True)
>>> torch.allclose(A @ X, B)
True

>>> A = torch.randn(2, 3, 3).tril_()
>>> B = torch.randn(2, 3, 4)
>>> X = torch.linalg.solve_triangular(A, B, upper=False)
>>> torch.allclose(A @ X, B)
True

>>> A = torch.randn(2, 4, 4).tril_()
>>> B = torch.randn(2, 3, 4)
>>> X = torch.linalg.solve_triangular(A, B, upper=False, left=False)
>>> torch.allclose(X @ A, B)
True

```

