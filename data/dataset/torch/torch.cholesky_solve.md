torch.cholesky_solve 
=============================================================================

torch. cholesky_solve ( *B*  , *L*  , *upper = False*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the solution of a system of linear equations with complex Hermitian
or real symmetric positive-definite lhs given its Cholesky decomposition. 

Let <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  be a complex Hermitian or real symmetric positive-definite matrix,
and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            L
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           L
          </annotation>
</semantics>
</math> -->L LL  its Cholesky decomposition such that: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            A
           </mi>
<mo>
            =
           </mo>
<mi>
            L
           </mi>
<msup>
<mi>
             L
            </mi>
<mtext>
             H
            </mtext>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           A = LL^{text{H}}
          </annotation>
</semantics>
</math> -->
A = L L H A = LL^{text{H}}

A = L L H

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
             L
            </mi>
<mtext>
             H
            </mtext>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           L^{text{H}}
          </annotation>
</semantics>
</math> -->L H L^{text{H}}L H  is the conjugate transpose when <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            L
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           L
          </annotation>
</semantics>
</math> -->L LL  is complex,
and the transpose when <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            L
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           L
          </annotation>
</semantics>
</math> -->L LL  is real-valued. 

Returns the solution <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            X
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           X
          </annotation>
</semantics>
</math> -->X XX  of the following linear system: 

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

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            B
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           B
          </annotation>
</semantics>
</math> -->B BB  is a batch of matrices
then the output has the same batch dimensions. 

Parameters
:   * **B** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – right-hand side tensor of shape *(*, n, k)* where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->∗ *∗  is zero or more batch dimensions

* **L** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, n, n)* where *** is zero or more batch dimensions
consisting of lower or upper triangular Cholesky decompositions of
symmetric or Hermitian positive-definite matrices.
* **upper** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – flag that indicates whether <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                L
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               L
              </annotation>
</semantics>
</math> -->L LL  is lower triangular
or upper triangular. Default: `False`  .

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – output tensor. Ignored if *None* . Default: *None* .

Example: 

```
>>> A = torch.randn(3, 3)
>>> A = A @ A.T + torch.eye(3) * 1e-3 # Creates a symmetric positive-definite matrix
>>> L = torch.linalg.cholesky(A) # Extract Cholesky decomposition
>>> B = torch.randn(3, 2)
>>> torch.cholesky_solve(B, L)
tensor([[ -8.1625,  19.6097],
        [ -5.8398,  14.2387],
        [ -4.3771,  10.4173]])
>>> A.inverse() @  B
tensor([[ -8.1626,  19.6097],
        [ -5.8398,  14.2387],
        [ -4.3771,  10.4173]])

>>> A = torch.randn(3, 2, 2, dtype=torch.complex64)
>>> A = A @ A.mH + torch.eye(2) * 1e-3 # Batch of Hermitian positive-definite matrices
>>> L = torch.linalg.cholesky(A)
>>> B = torch.randn(2, 1, dtype=torch.complex64)
>>> X = torch.cholesky_solve(B, L)
>>> torch.dist(X, A.inverse() @ B)
tensor(1.6881e-5)

```

