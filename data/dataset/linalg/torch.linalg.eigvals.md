torch.linalg.eigvals 
============================================================================

torch.linalg. eigvals ( *A*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the eigenvalues of a square matrix. 

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
the **eigenvalues** of a square matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A ∈ K n × n A in mathbb{K}^{n times n}A ∈ K n × n  are defined
as the roots (counted with multiplicity) of the polynomial *p* of degree *n* given by 

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
              C
             </mi>
</mrow>
</mpadded>
</mrow>
<annotation encoding="application/x-tex">
           p(lambda) = operatorname{det}(A - lambda mathrm{I}_n)mathrlap{qquad lambda in mathbb{C}}
          </annotation>
</semantics>
</math> -->
p ( λ ) = det ⁡ ( A − λ I n ) λ ∈ C p(lambda) = operatorname{det}(A - lambda mathrm{I}_n)mathrlap{qquad lambda in mathbb{C}}

p ( λ ) = det ( A − λ I n ​ ) λ ∈ C

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

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if `A`  is a batch of matrices then
the output has the same batch dimensions. 

The returned eigenvalues are not guaranteed to be in any specific order. 

Note 

The eigenvalues of a real matrix may be complex, as the roots of a real polynomial may be complex. 

The eigenvalues of a matrix are always well-defined, even when the matrix is not diagonalizable.

Note 

When inputs are on a CUDA device, this function synchronizes that device with the CPU.

See also 

[`torch.linalg.eig()`](torch.linalg.eig.html#torch.linalg.eig "torch.linalg.eig")  computes the full eigenvalue decomposition.

Parameters
: **A** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, n, n)* where *** is zero or more batch dimensions.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – output tensor. Ignored if *None* . Default: *None* .

Returns
:   A complex-valued tensor containing the eigenvalues even when `A`  is real.

Examples: 

```
>>> A = torch.randn(2, 2, dtype=torch.complex128)
>>> L = torch.linalg.eigvals(A)
>>> L
tensor([ 1.1226+0.5738j, -0.7537-0.1286j], dtype=torch.complex128)

>>> torch.dist(L, torch.linalg.eig(A).eigenvalues)
tensor(2.4576e-07)

```

