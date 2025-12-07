torch.cholesky_inverse 
=================================================================================

torch. cholesky_inverse ( *L*  , *upper = False*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the inverse of a complex Hermitian or real symmetric
positive-definite matrix given its Cholesky decomposition. 

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

Computes the inverse matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
             A
            </mi>
<mrow>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           A^{-1}
          </annotation>
</semantics>
</math> -->A − 1 A^{-1}A − 1  . 

Supports input of float, double, cfloat and cdouble dtypes.
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
</math> -->A AA  is a batch of matrices
then the output has the same batch dimensions. 

Parameters
:   * **L** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, n, n)* where *** is zero or more batch dimensions
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
or upper triangular. Default: `False`

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – output tensor. Ignored if *None* . Default: *None* .

Example: 

```
>>> A = torch.randn(3, 3)
>>> A = A @ A.T + torch.eye(3) * 1e-3 # Creates a symmetric positive-definite matrix
>>> L = torch.linalg.cholesky(A) # Extract Cholesky decomposition
>>> torch.cholesky_inverse(L)
tensor([[ 1.9314,  1.2251, -0.0889],
        [ 1.2251,  2.4439,  0.2122],
        [-0.0889,  0.2122,  0.1412]])
>>> A.inverse()
tensor([[ 1.9314,  1.2251, -0.0889],
        [ 1.2251,  2.4439,  0.2122],
        [-0.0889,  0.2122,  0.1412]])

>>> A = torch.randn(3, 2, 2, dtype=torch.complex64)
>>> A = A @ A.mH + torch.eye(2) * 1e-3 # Batch of Hermitian positive-definite matrices
>>> L = torch.linalg.cholesky(A)
>>> torch.dist(torch.inverse(A), torch.cholesky_inverse(L))
tensor(5.6358e-7)

```

