torch.linalg.lu 
==================================================================

torch.linalg. lu ( *A*  , *** , *pivot = True*  , *out = None* ) 
:   Computes the LU decomposition with partial pivoting of a matrix. 

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
the **LU decomposition with partial pivoting** of a matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
              m
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
           A in mathbb{K}^{m times n}
          </annotation>
</semantics>
</math> -->A ∈ K m × n A in mathbb{K}^{m times n}A ∈ K m × n  is defined as 

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
            P
           </mi>
<mi>
            L
           </mi>
<mi>
            U
           </mi>
<mpadded width="0px">
<mrow>
<mspace width="2em">
</mspace>
<mi>
              P
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
                m
               </mi>
<mo>
                ×
               </mo>
<mi>
                m
               </mi>
</mrow>
</msup>
<mo separator="true">
              ,
             </mo>
<mi>
              L
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
                m
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
              U
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
                n
               </mi>
</mrow>
</msup>
</mrow>
</mpadded>
</mrow>
<annotation encoding="application/x-tex">
           A = PLUmathrlap{qquad P in mathbb{K}^{m times m}, L in mathbb{K}^{m times k}, U in mathbb{K}^{k times n}}
          </annotation>
</semantics>
</math> -->
A = P L U P ∈ K m × m , L ∈ K m × k , U ∈ K k × n A = PLUmathrlap{qquad P in mathbb{K}^{m times m}, L in mathbb{K}^{m times k}, U in mathbb{K}^{k times n}}

A = P LU P ∈ K m × m , L ∈ K m × k , U ∈ K k × n

where *k = min(m,n)* , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            P
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           P
          </annotation>
</semantics>
</math> -->P PP  is a [permutation matrix](https://en.wikipedia.org/wiki/Permutation_matrix)  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->L LL  is lower triangular with ones on the diagonal
and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            U
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           U
          </annotation>
</semantics>
</math> -->U UU  is upper triangular. 

If `pivot` *= False* and `A`  is on GPU, then the **LU decomposition without pivoting** is computed 

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
<mi>
            U
           </mi>
<mpadded width="0px">
<mrow>
<mspace width="2em">
</mspace>
<mi>
              L
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
                m
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
              U
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
                n
               </mi>
</mrow>
</msup>
</mrow>
</mpadded>
</mrow>
<annotation encoding="application/x-tex">
           A = LUmathrlap{qquad L in mathbb{K}^{m times k}, U in mathbb{K}^{k times n}}
          </annotation>
</semantics>
</math> -->
A = L U L ∈ K m × k , U ∈ K k × n A = LUmathrlap{qquad L in mathbb{K}^{m times k}, U in mathbb{K}^{k times n}}

A = LU L ∈ K m × k , U ∈ K k × n

When `pivot` *= False* , the returned matrix `P`  will be empty.
The LU decomposition without pivoting [may not exist](https://en.wikipedia.org/wiki/LU_decomposition#Definitions)  if any of the principal minors of `A`  is singular.
In this case, the output matrix may contain *inf* or *NaN* . 

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if `A`  is a batch of matrices then
the output has the same batch dimensions. 

See also 

[`torch.linalg.solve()`](torch.linalg.solve.html#torch.linalg.solve "torch.linalg.solve")  solves a system of linear equations using the LU decomposition
with partial pivoting.

Warning 

The LU decomposition is almost never unique, as often there are different permutation
matrices that can yield different LU decompositions.
As such, different platforms, like SciPy, or inputs on different devices,
may produce different valid decompositions.

Warning 

Gradient computations are only supported if the input matrix is full-rank.
If this condition is not met, no error will be thrown, but the gradient
may not be finite.
This is because the LU decomposition with pivoting is not differentiable at these points.

Parameters
:   * **A** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, m, n)* where *** is zero or more batch dimensions.
* **pivot** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Controls whether to compute the LU decomposition with partial pivoting or
no pivoting. Default: *True* .

Keyword Arguments
: **out** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – output tuple of three tensors. Ignored if *None* . Default: *None* .

Returns
:   A named tuple *(P, L, U)* .

Examples: 

```
>>> A = torch.randn(3, 2)
>>> P, L, U = torch.linalg.lu(A)
>>> P
tensor([[0., 1., 0.],
        [0., 0., 1.],
        [1., 0., 0.]])
>>> L
tensor([[1.0000, 0.0000],
        [0.5007, 1.0000],
        [0.0633, 0.9755]])
>>> U
tensor([[0.3771, 0.0489],
        [0.0000, 0.9644]])
>>> torch.dist(A, P @ L @ U)
tensor(5.9605e-08)

>>> A = torch.randn(2, 5, 7, device="cuda")
>>> P, L, U = torch.linalg.lu(A, pivot=False)
>>> P
tensor([], device='cuda:0')
>>> torch.dist(A, L @ U)
tensor(1.0376e-06, device='cuda:0')

```

