torch.linalg.cond 
======================================================================

torch.linalg. cond ( *A*  , *p = None*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the condition number of a matrix with respect to a matrix norm. 

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
the **condition number** <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            κ
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           kappa
          </annotation>
</semantics>
</math> -->κ kappaκ  of a matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
            κ
           </mi>
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
<mi mathvariant="normal">
            ∥
           </mi>
<mi>
            A
           </mi>
<msub>
<mi mathvariant="normal">
             ∥
            </mi>
<mi>
             p
            </mi>
</msub>
<mi mathvariant="normal">
            ∥
           </mi>
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
<msub>
<mi mathvariant="normal">
             ∥
            </mi>
<mi>
             p
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           kappa(A) = |A|_p|A^{-1}|_p
          </annotation>
</semantics>
</math> -->
κ ( A ) = ∥ A ∥ p ∥ A − 1 ∥ p kappa(A) = |A|_p|A^{-1}|_p

κ ( A ) = ∥ A ∥ p ​ ∥ A − 1 ∥ p ​

The condition number of `A`  measures the numerical stability of the linear system *AX = B* with respect to a matrix norm. 

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if `A`  is a batch of matrices then
the output has the same batch dimensions. 

`p`  defines the matrix norm that is computed. The following norms are supported: 

| `p` | matrix norm |
| --- | --- |
| None | 2  -norm (largest singular value) |
| ‘fro’ | Frobenius norm |
| ‘nuc’ | nuclear norm |
| inf | max(sum(abs(x), dim=1)) |
| -inf | min(sum(abs(x), dim=1)) |
| 1 | max(sum(abs(x), dim=0)) |
| -1 | min(sum(abs(x), dim=0)) |
| 2 | largest singular value |
| -2 | smallest singular value |

where *inf* refers to *float(‘inf’)* , NumPy’s *inf* object, or any equivalent object. 

For `p`  is one of *(‘fro’, ‘nuc’, inf, -inf, 1, -1)* , this function uses [`torch.linalg.norm()`](torch.linalg.norm.html#torch.linalg.norm "torch.linalg.norm")  and [`torch.linalg.inv()`](torch.linalg.inv.html#torch.linalg.inv "torch.linalg.inv")  .
As such, in this case, the matrix (or every matrix in the batch) `A`  has to be square
and invertible. 

For `p`  in *(2, -2)* , this function can be computed in terms of the singular values <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             σ
            </mi>
<mn>
             1
            </mn>
</msub>
<mo>
            ≥
           </mo>
<mo>
            …
           </mo>
<mo>
            ≥
           </mo>
<msub>
<mi>
             σ
            </mi>
<mi>
             n
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           sigma_1 geq ldots geq sigma_n
          </annotation>
</semantics>
</math> -->σ 1 ≥ … ≥ σ n sigma_1 geq ldots geq sigma_nσ 1 ​ ≥ … ≥ σ n ​ 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             κ
            </mi>
<mn>
             2
            </mn>
</msub>
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
<mfrac>
<msub>
<mi>
              σ
             </mi>
<mn>
              1
             </mn>
</msub>
<msub>
<mi>
              σ
             </mi>
<mi>
              n
             </mi>
</msub>
</mfrac>
<mspace width="2em">
</mspace>
<msub>
<mi>
             κ
            </mi>
<mrow>
<mo>
              −
             </mo>
<mn>
              2
             </mn>
</mrow>
</msub>
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
<mfrac>
<msub>
<mi>
              σ
             </mi>
<mi>
              n
             </mi>
</msub>
<msub>
<mi>
              σ
             </mi>
<mn>
              1
             </mn>
</msub>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           kappa_2(A) = frac{sigma_1}{sigma_n}qquad kappa_{-2}(A) = frac{sigma_n}{sigma_1}
          </annotation>
</semantics>
</math> -->
κ 2 ( A ) = σ 1 σ n κ − 2 ( A ) = σ n σ 1 kappa_2(A) = frac{sigma_1}{sigma_n}qquad kappa_{-2}(A) = frac{sigma_n}{sigma_1}

κ 2 ​ ( A ) = σ n ​ σ 1 ​ ​ κ − 2 ​ ( A ) = σ 1 ​ σ n ​ ​

In these cases, it is computed using [`torch.linalg.svdvals()`](torch.linalg.svdvals.html#torch.linalg.svdvals "torch.linalg.svdvals")  . For these norms, the matrix
(or every matrix in the batch) `A`  may have any shape. 

Note 

When inputs are on a CUDA device, this function synchronizes that device with the CPU
if `p`  is one of *(‘fro’, ‘nuc’, inf, -inf, 1, -1)* .

See also 

[`torch.linalg.solve()`](torch.linalg.solve.html#torch.linalg.solve "torch.linalg.solve")  for a function that solves linear systems of square matrices. 

[`torch.linalg.lstsq()`](torch.linalg.lstsq.html#torch.linalg.lstsq "torch.linalg.lstsq")  for a function that solves linear systems of general matrices.

Parameters
:   * **A** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, m, n)* where *** is zero or more batch dimensions
for `p`  in *(2, -2)* , and of shape *(*, n, n)* where every matrix
is invertible for `p`  in *(‘fro’, ‘nuc’, inf, -inf, 1, -1)* .
* **p** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *inf* *,* *-inf* *,* *'fro'* *,* *'nuc'* *,* *optional*  ) – the type of the matrix norm to use in the computations (see above). Default: *None*

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – output tensor. Ignored if *None* . Default: *None* .

Returns
:   A real-valued tensor, even when `A`  is complex.

Raises
:   [**RuntimeError**](https://docs.python.org/3/library/exceptions.html#RuntimeError "(in Python v3.13)")  – if `p`  is one of *(‘fro’, ‘nuc’, inf, -inf, 1, -1)* and the `A`  matrix or any matrix in the batch `A`  is not square
 or invertible.

Examples: 

```
>>> A = torch.randn(3, 4, 4, dtype=torch.complex64)
>>> torch.linalg.cond(A)
>>> A = torch.tensor([[1., 0, -1], [0, 1, 0], [1, 0, 1]])
>>> torch.linalg.cond(A)
tensor([1.4142])
>>> torch.linalg.cond(A, 'fro')
tensor(3.1623)
>>> torch.linalg.cond(A, 'nuc')
tensor(9.2426)
>>> torch.linalg.cond(A, float('inf'))
tensor(2.)
>>> torch.linalg.cond(A, float('-inf'))
tensor(1.)
>>> torch.linalg.cond(A, 1)
tensor(2.)
>>> torch.linalg.cond(A, -1)
tensor(1.)
>>> torch.linalg.cond(A, 2)
tensor([1.4142])
>>> torch.linalg.cond(A, -2)
tensor([0.7071])

>>> A = torch.randn(2, 3, 3)
>>> torch.linalg.cond(A)
tensor([[9.5917],
        [3.2538]])
>>> A = torch.randn(2, 3, 3, dtype=torch.complex64)
>>> torch.linalg.cond(A)
tensor([[4.6245],
        [4.5671]])

```

