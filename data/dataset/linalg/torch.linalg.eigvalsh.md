torch.linalg.eigvalsh 
==============================================================================

torch.linalg. eigvalsh ( *A*  , *UPLO = 'L'*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the eigenvalues of a complex Hermitian or real symmetric matrix. 

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
the **eigenvalues** of a complex Hermitian or real symmetric matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A ∈ K n × n A in mathbb{K}^{n times n}A ∈ K n × n  are defined as the roots (counted with multiplicity) of the polynomial *p* of degree *n* given by 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            p
           </mi>
<mo stretchy="false">
            (
           </mo>
<mi>
            λ
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mi mathvariant="normal">
            det
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<mi>
            A
           </mi>
<mo>
            −
           </mo>
<mi>
            λ
           </mi>
<msub>
<mi mathvariant="normal">
             I
            </mi>
<mi>
             n
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
<mpadded width="0px">
<mrow>
<mspace width="2em">
</mspace>
<mi>
              λ
             </mi>
<mo>
              ∈
             </mo>
<mi mathvariant="double-struck">
              R
             </mi>
</mrow>
</mpadded>
</mrow>
<annotation encoding="application/x-tex">
           p(lambda) = operatorname{det}(A - lambda mathrm{I}_n)mathrlap{qquad lambda in mathbb{R}}
          </annotation>
</semantics>
</math> -->
p ( λ ) = det ⁡ ( A − λ I n ) λ ∈ R p(lambda) = operatorname{det}(A - lambda mathrm{I}_n)mathrlap{qquad lambda in mathbb{R}}

p ( λ ) = det ( A − λ I n ​ ) λ ∈ R

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi mathvariant="normal">
             I
            </mi>
<mi>
             n
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           mathrm{I}_n
          </annotation>
</semantics>
</math> -->I n mathrm{I}_nI n ​  is the *n* -dimensional identity matrix.
The eigenvalues of a real symmetric or complex Hermitian matrix are always real. 

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if `A`  is a batch of matrices then
the output has the same batch dimensions. 

The eigenvalues are returned in ascending order. 

`A`  is assumed to be Hermitian (resp. symmetric), but this is not checked internally, instead: 

* If `UPLO` *= ‘L’* (default), only the lower triangular part of the matrix is used in the computation.
* If `UPLO` *= ‘U’* , only the upper triangular part of the matrix is used.

Note 

When inputs are on a CUDA device, this function synchronizes that device with the CPU.

See also 

[`torch.linalg.eigh()`](torch.linalg.eigh.html#torch.linalg.eigh "torch.linalg.eigh")  computes the full eigenvalue decomposition.

Parameters
:   * **A** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, n, n)* where *** is zero or more batch dimensions
consisting of symmetric or Hermitian matrices.
* **UPLO** ( *'L'* *,* *'U'* *,* *optional*  ) – controls whether to use the upper or lower triangular part
of `A`  in the computations. Default: *‘L’* .

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – output tensor. Ignored if *None* . Default: *None* .

Returns
:   A real-valued tensor containing the eigenvalues even when `A`  is complex.
The eigenvalues are returned in ascending order.

Examples: 

```
>>> A = torch.randn(2, 2, dtype=torch.complex128)
>>> A = A + A.T.conj()  # creates a Hermitian matrix
>>> A
tensor([[2.9228+0.0000j, 0.2029-0.0862j],
        [0.2029+0.0862j, 0.3464+0.0000j]], dtype=torch.complex128)
>>> torch.linalg.eigvalsh(A)
tensor([0.3277, 2.9415], dtype=torch.float64)

>>> A = torch.randn(3, 2, 2, dtype=torch.float64)
>>> A = A + A.mT  # creates a batch of symmetric matrices
>>> torch.linalg.eigvalsh(A)
tensor([[ 2.5797,  3.4629],
        [-4.1605,  1.3780],
        [-3.1113,  2.7381]], dtype=torch.float64)

```

