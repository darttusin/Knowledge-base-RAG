torch.linalg.eig 
====================================================================

torch.linalg. eig ( *A*  , *** , *out = None* ) 
:   Computes the eigenvalue decomposition of a square matrix if it exists. 

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
the **eigenvalue decomposition** of a square matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A ∈ K n × n A in mathbb{K}^{n times n}A ∈ K n × n  (if it exists) is defined as 

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
            V
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
             V
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
<mpadded width="0px">
<mrow>
<mspace width="2em">
</mspace>
<mi>
              V
             </mi>
<mo>
              ∈
             </mo>
<msup>
<mi mathvariant="double-struck">
               C
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
               C
              </mi>
<mi>
               n
              </mi>
</msup>
</mrow>
</mpadded>
</mrow>
<annotation encoding="application/x-tex">
           A = V operatorname{diag}(Lambda) V^{-1}mathrlap{qquad V in mathbb{C}^{n times n}, Lambda in mathbb{C}^n}
          </annotation>
</semantics>
</math> -->
A = V diag ⁡ ( Λ ) V − 1 V ∈ C n × n , Λ ∈ C n A = V operatorname{diag}(Lambda) V^{-1}mathrlap{qquad V in mathbb{C}^{n times n}, Lambda in mathbb{C}^n}

A = V diag ( Λ ) V − 1 V ∈ C n × n , Λ ∈ C n

This decomposition exists if and only if <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  is [diagonalizable](https://en.wikipedia.org/wiki/Diagonalizable_matrix#Definition)  .
This is the case when all its eigenvalues are different. 

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if `A`  is a batch of matrices then
the output has the same batch dimensions. 

The returned eigenvalues are not guaranteed to be in any specific order. 

Note 

The eigenvalues and eigenvectors of a real matrix may be complex.

Note 

When inputs are on a CUDA device, this function synchronizes that device with the CPU.

Warning 

This function assumes that `A`  is [diagonalizable](https://en.wikipedia.org/wiki/Diagonalizable_matrix#Definition)  (for example, when all the
eigenvalues are different). If it is not diagonalizable, the returned
eigenvalues will be correct but <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             A
            </mi>
<mo mathvariant="normal">
             ≠
            </mo>
<mi>
             V
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
              V
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
            A neq V operatorname{diag}(Lambda)V^{-1}
           </annotation>
</semantics>
</math> -->A ≠ V diag ⁡ ( Λ ) V − 1 A neq V operatorname{diag}(Lambda)V^{-1}A  = V diag ( Λ ) V − 1  .

Warning 

The returned eigenvectors are normalized to have norm *1* .
Even then, the eigenvectors of a matrix are not unique, nor are they continuous with respect to `A`  . Due to this lack of uniqueness, different hardware and software may compute
different eigenvectors. 

This non-uniqueness is caused by the fact that multiplying an eigenvector by
by <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->e i ϕ , ϕ ∈ R e^{i phi}, phi in mathbb{R}e i ϕ , ϕ ∈ R  produces another set of valid eigenvectors
of the matrix. For this reason, the loss function shall not depend on the phase of the
eigenvectors, as this quantity is not well-defined.
This is checked when computing the gradients of this function. As such,
when inputs are on a CUDA device, the computation of the gradients
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

See also 

[`torch.linalg.eigvals()`](torch.linalg.eigvals.html#torch.linalg.eigvals "torch.linalg.eigvals")  computes only the eigenvalues.
Unlike [`torch.linalg.eig()`](#torch.linalg.eig "torch.linalg.eig")  , the gradients of [`eigvals()`](torch.linalg.eigvals.html#torch.linalg.eigvals "torch.linalg.eigvals")  are always
numerically stable. 

[`torch.linalg.eigh()`](torch.linalg.eigh.html#torch.linalg.eigh "torch.linalg.eigh")  for a (faster) function that computes the eigenvalue decomposition
for Hermitian and symmetric matrices. 

[`torch.linalg.svd()`](torch.linalg.svd.html#torch.linalg.svd "torch.linalg.svd")  for a function that computes another type of spectral
decomposition that works on matrices of any shape. 

[`torch.linalg.qr()`](torch.linalg.qr.html#torch.linalg.qr "torch.linalg.qr")  for another (much faster) decomposition that works on matrices of
any shape.

Parameters
: **A** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, n, n)* where *** is zero or more batch dimensions
consisting of diagonalizable matrices.

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
              V
             </mi>
</mrow>
<annotation encoding="application/x-tex">
             V
            </annotation>
</semantics>
</math> -->V VV  above. 

*eigenvalues* and *eigenvectors* will always be complex-valued, even when `A`  is real. The eigenvectors
will be given by the columns of *eigenvectors* .

Examples: 

```
>>> A = torch.randn(2, 2, dtype=torch.complex128)
>>> A
tensor([[ 0.9828+0.3889j, -0.4617+0.3010j],
        [ 0.1662-0.7435j, -0.6139+0.0562j]], dtype=torch.complex128)
>>> L, V = torch.linalg.eig(A)
>>> L
tensor([ 1.1226+0.5738j, -0.7537-0.1286j], dtype=torch.complex128)
>>> V
tensor([[ 0.9218+0.0000j,  0.1882-0.2220j],
        [-0.0270-0.3867j,  0.9567+0.0000j]], dtype=torch.complex128)
>>> torch.dist(V @ torch.diag(L) @ torch.linalg.inv(V), A)
tensor(7.7119e-16, dtype=torch.float64)

>>> A = torch.randn(3, 2, 2, dtype=torch.float64)
>>> L, V = torch.linalg.eig(A)
>>> torch.dist(V @ torch.diag_embed(L) @ torch.linalg.inv(V), A)
tensor(3.2841e-16, dtype=torch.float64)

```

