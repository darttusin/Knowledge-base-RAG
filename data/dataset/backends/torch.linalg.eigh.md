torch.linalg.eigh 
======================================================================

torch.linalg. eigh ( *A*  , *UPLO = 'L'*  , *** , *out = None* ) 
:   Computes the eigenvalue decomposition of a complex Hermitian or real symmetric matrix. 

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
the **eigenvalue decomposition** of a complex Hermitian or real symmetric matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A ∈ K n × n A in mathbb{K}^{n times n}A ∈ K n × n  is defined as 

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
            Q
           </mi>
<mi mathvariant="normal">
            diag
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<mi mathvariant="normal">
            Λ
           </mi>
<mo stretchy="false">
            )
           </mo>
<msup>
<mi>
             Q
            </mi>
<mtext>
             H
            </mtext>
</msup>
<mpadded width="0px">
<mrow>
<mspace width="2em">
</mspace>
<mi>
              Q
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
<mo separator="true">
              ,
             </mo>
<mi mathvariant="normal">
              Λ
             </mi>
<mo>
              ∈
             </mo>
<msup>
<mi mathvariant="double-struck">
               R
              </mi>
<mi>
               n
              </mi>
</msup>
</mrow>
</mpadded>
</mrow>
<annotation encoding="application/x-tex">
           A = Q operatorname{diag}(Lambda) Q^{text{H}}mathrlap{qquad Q in mathbb{K}^{n times n}, Lambda in mathbb{R}^n}
          </annotation>
</semantics>
</math> -->
A = Q diag ⁡ ( Λ ) Q H Q ∈ K n × n , Λ ∈ R n A = Q operatorname{diag}(Lambda) Q^{text{H}}mathrlap{qquad Q in mathbb{K}^{n times n}, Lambda in mathbb{R}^n}

A = Q diag ( Λ ) Q H Q ∈ K n × n , Λ ∈ R n

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
             Q
            </mi>
<mtext>
             H
            </mtext>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           Q^{text{H}}
          </annotation>
</semantics>
</math> -->Q H Q^{text{H}}Q H  is the conjugate transpose when <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            Q
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           Q
          </annotation>
</semantics>
</math> -->Q QQ  is complex, and the transpose when <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            Q
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           Q
          </annotation>
</semantics>
</math> -->Q QQ  is real-valued. <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            Q
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           Q
          </annotation>
</semantics>
</math> -->Q QQ  is orthogonal in the real case and unitary in the complex case. 

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if `A`  is a batch of matrices then
the output has the same batch dimensions. 

`A`  is assumed to be Hermitian (resp. symmetric), but this is not checked internally, instead: 

* If `UPLO` *= ‘L’* (default), only the lower triangular part of the matrix is used in the computation.
* If `UPLO` *= ‘U’* , only the upper triangular part of the matrix is used.

The eigenvalues are returned in ascending order. 

Note 

When inputs are on a CUDA device, this function synchronizes that device with the CPU.

Note 

The eigenvalues of real symmetric or complex Hermitian matrices are always real.

Warning 

The eigenvectors of a symmetric matrix are not unique, nor are they continuous with
respect to `A`  . Due to this lack of uniqueness, different hardware and
software may compute different eigenvectors. 

This non-uniqueness is caused by the fact that multiplying an eigenvector by *-1* in the real case or by <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
              e
             </mi>
<mrow>
<mi>
               i
              </mi>
<mi>
               ϕ
              </mi>
</mrow>
</msup>
<mo separator="true">
             ,
            </mo>
<mi>
             ϕ
            </mi>
<mo>
             ∈
            </mo>
<mi mathvariant="double-struck">
             R
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            e^{i phi}, phi in mathbb{R}
           </annotation>
</semantics>
</math> -->e i ϕ , ϕ ∈ R e^{i phi}, phi in mathbb{R}e i ϕ , ϕ ∈ R  in the complex
case produces another set of valid eigenvectors of the matrix.
For this reason, the loss function shall not depend on the phase of the eigenvectors, as
this quantity is not well-defined.
This is checked for complex inputs when computing the gradients of this function. As such,
when inputs are complex and are on a CUDA device, the computation of the gradients
of this function synchronizes that device with the CPU.

Warning 

Gradients computed using the *eigenvectors* tensor will only be finite when `A`  has distinct eigenvalues.
Furthermore, if the distance between any two eigenvalues is close to zero,
the gradient will be numerically unstable, as it depends on the eigenvalues <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</mrow>
<annotation encoding="application/x-tex">
            lambda_i
           </annotation>
</semantics>
</math> -->λ i lambda_iλ i ​  through the computation of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mfrac>
<mn>
              1
             </mn>
<mrow>
<msub>
<mrow>
<mi>
                 min
                </mi>
<mo>
                 ⁡
                </mo>
</mrow>
<mrow>
<mi>
                 i
                </mi>
<mo mathvariant="normal">
                 ≠
                </mo>
<mi>
                 j
                </mi>
</mrow>
</msub>
<msub>
<mi>
                λ
               </mi>
<mi>
                i
               </mi>
</msub>
<mo>
               −
              </mo>
<msub>
<mi>
                λ
               </mi>
<mi>
                j
               </mi>
</msub>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
            frac{1}{min_{i neq j} lambda_i - lambda_j}
           </annotation>
</semantics>
</math> -->1 min ⁡ i ≠ j λ i − λ j frac{1}{min_{i neq j} lambda_i - lambda_j}m i n i  = j ​ λ i ​ − λ j ​ 1 ​  .

Warning 

User may see pytorch crashes if running *eigh* on CUDA devices with CUDA versions before 12.1 update 1
with large ill-conditioned matrices as inputs.
Refer to [Linear Algebra Numerical Stability](../notes/numerical_accuracy.html#linear-algebra-stability)  for more details.
If this is the case, user may (1) tune their matrix inputs to be less ill-conditioned,
or (2) use [`torch.backends.cuda.preferred_linalg_library()`](../backends.html#torch.backends.cuda.preferred_linalg_library "torch.backends.cuda.preferred_linalg_library")  to
try other supported backends.

See also 

[`torch.linalg.eigvalsh()`](torch.linalg.eigvalsh.html#torch.linalg.eigvalsh "torch.linalg.eigvalsh")  computes only the eigenvalues of a Hermitian matrix.
Unlike [`torch.linalg.eigh()`](#torch.linalg.eigh "torch.linalg.eigh")  , the gradients of [`eigvalsh()`](torch.linalg.eigvalsh.html#torch.linalg.eigvalsh "torch.linalg.eigvalsh")  are always
numerically stable. 

[`torch.linalg.cholesky()`](torch.linalg.cholesky.html#torch.linalg.cholesky "torch.linalg.cholesky")  for a different decomposition of a Hermitian matrix.
The Cholesky decomposition gives less information about the matrix but is much faster
to compute than the eigenvalue decomposition. 

[`torch.linalg.eig()`](torch.linalg.eig.html#torch.linalg.eig "torch.linalg.eig")  for a (slower) function that computes the eigenvalue decomposition
of a not necessarily Hermitian square matrix. 

[`torch.linalg.svd()`](torch.linalg.svd.html#torch.linalg.svd "torch.linalg.svd")  for a (slower) function that computes the more general SVD
decomposition of matrices of any shape. 

[`torch.linalg.qr()`](torch.linalg.qr.html#torch.linalg.qr "torch.linalg.qr")  for another (much faster) decomposition that works on general
matrices.

Parameters
:   * **A** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, n, n)* where *** is zero or more batch dimensions
consisting of symmetric or Hermitian matrices.
* **UPLO** ( *'L'* *,* *'U'* *,* *optional*  ) – controls whether to use the upper or lower triangular part
of `A`  in the computations. Default: *‘L’* .

Keyword Arguments
: **out** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – output tuple of two tensors. Ignored if *None* . Default: *None* .

Returns
:   A named tuple *(eigenvalues, eigenvectors)* which corresponds to <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
              Λ
             </mi>
</mrow>
<annotation encoding="application/x-tex">
             Lambda
            </annotation>
</semantics>
</math> -->Λ LambdaΛ  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
              Q
             </mi>
</mrow>
<annotation encoding="application/x-tex">
             Q
            </annotation>
</semantics>
</math> -->Q QQ  above. 

*eigenvalues* will always be real-valued, even when `A`  is complex.
It will also be ordered in ascending order. 

*eigenvectors* will have the same dtype as `A`  and will contain the eigenvectors as its columns.

Examples::
:   ```
>>> A = torch.randn(2, 2, dtype=torch.complex128)
>>> A = A + A.T.conj()  # creates a Hermitian matrix
>>> A
tensor([[2.9228+0.0000j, 0.2029-0.0862j],
        [0.2029+0.0862j, 0.3464+0.0000j]], dtype=torch.complex128)
>>> L, Q = torch.linalg.eigh(A)
>>> L
tensor([0.3277, 2.9415], dtype=torch.float64)
>>> Q
tensor([[-0.0846+-0.0000j, -0.9964+0.0000j],
        [ 0.9170+0.3898j, -0.0779-0.0331j]], dtype=torch.complex128)
>>> torch.dist(Q @ torch.diag(L.cdouble()) @ Q.T.conj(), A)
tensor(6.1062e-16, dtype=torch.float64)

```

```
>>> A = torch.randn(3, 2, 2, dtype=torch.float64)
>>> A = A + A.mT  # creates a batch of symmetric matrices
>>> L, Q = torch.linalg.eigh(A)
>>> torch.dist(Q @ torch.diag_embed(L) @ Q.mH, A)
tensor(1.5423e-15, dtype=torch.float64)

```

