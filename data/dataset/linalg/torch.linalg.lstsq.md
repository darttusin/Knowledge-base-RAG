torch.linalg.lstsq 
========================================================================

torch.linalg. lstsq ( *A*  , *B*  , *rcond = None*  , *** , *driver = None* ) 
:   Computes a solution to the least squares problem of a system of linear equations. 

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
the **least squares problem** for a linear system <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A X = B AX = BA X = B  with <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</mrow>
<annotation encoding="application/x-tex">
           A in mathbb{K}^{m times n}, B in mathbb{K}^{m times k}
          </annotation>
</semantics>
</math> -->A ∈ K m × n , B ∈ K m × k A in mathbb{K}^{m times n}, B in mathbb{K}^{m times k}A ∈ K m × n , B ∈ K m × k  is defined as 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<munder>
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
</munder>
<mi mathvariant="normal">
            ∥
           </mi>
<mi>
            A
           </mi>
<mi>
            X
           </mi>
<mo>
            −
           </mo>
<mi>
            B
           </mi>
<msub>
<mi mathvariant="normal">
             ∥
            </mi>
<mi>
             F
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           min_{X in mathbb{K}^{n times k}} |AX - B|_F
          </annotation>
</semantics>
</math> -->
min ⁡ X ∈ K n × k ∥ A X − B ∥ F min_{X in mathbb{K}^{n times k}} |AX - B|_F

X ∈ K n × k min ​ ∥ A X − B ∥ F ​

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
            ∥
           </mi>
<mo>
            −
           </mo>
<msub>
<mi mathvariant="normal">
             ∥
            </mi>
<mi>
             F
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           |-|_F
          </annotation>
</semantics>
</math> -->∥ − ∥ F |-|_F∥ − ∥ F ​  denotes the Frobenius norm. 

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if the inputs are batches of matrices then
the output has the same batch dimensions. 

`driver`  chooses the backend function that will be used.
For CPU inputs the valid values are *‘gels’* , *‘gelsy’* , *‘gelsd* , *‘gelss’* .
To choose the best driver on CPU consider: 

* If `A`  is well-conditioned (its [condition number](https://localhost:8000/docs/main/linalg.html#torch.linalg.cond)  is not too large), or you do not mind some precision loss.

    + For a general matrix: *‘gelsy’* (QR with pivoting) (default)
        + If `A`  is full-rank: *‘gels’* (QR)
* If `A`  is not well-conditioned.

    + *‘gelsd’* (tridiagonal reduction and SVD)
        + But if you run into memory issues: *‘gelss’* (full SVD).

For CUDA input, the only valid driver is *‘gels’* , which assumes that `A`  is full-rank. 

See also the [full description of these drivers](https://www.netlib.org/lapack/lug/node27.html) 

`rcond`  is used to determine the effective rank of the matrices in `A`  when `driver`  is one of ( *‘gelsy’* , *‘gelsd’* , *‘gelss’* ).
In this case, if <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             σ
            </mi>
<mi>
             i
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           sigma_i
          </annotation>
</semantics>
</math> -->σ i sigma_iσ i ​  are the singular values of *A* in decreasing order, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             σ
            </mi>
<mi>
             i
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           sigma_i
          </annotation>
</semantics>
</math> -->σ i sigma_iσ i ​  will be rounded down to zero if <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             σ
            </mi>
<mi>
             i
            </mi>
</msub>
<mo>
            ≤
           </mo>
<mtext>
            rcond
           </mtext>
<mo>
            ⋅
           </mo>
<msub>
<mi>
             σ
            </mi>
<mn>
             1
            </mn>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           sigma_i leq text{rcond} cdot sigma_1
          </annotation>
</semantics>
</math> -->σ i ≤ rcond ⋅ σ 1 sigma_i leq text{rcond} cdot sigma_1σ i ​ ≤ rcond ⋅ σ 1 ​  .
If `rcond` *= None* (default), `rcond`  is set to the machine precision of the dtype of `A`  times *max(m, n)* . 

This function returns the solution to the problem and some extra information in a named tuple of
four tensors *(solution, residuals, rank, singular_values)* . For inputs `A`  , `B`  of shape *(*, m, n)* , *(*, m, k)* respectively, it contains 

* *solution* : the least squares solution. It has shape *(*, n, k)* .
* *residuals* : the squared residuals of the solutions, that is, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
              ∥
             </mi>
<mi>
              A
             </mi>
<mi>
              X
             </mi>
<mo>
              −
             </mo>
<mi>
              B
             </mi>
<msubsup>
<mi mathvariant="normal">
               ∥
              </mi>
<mi>
               F
              </mi>
<mn>
               2
              </mn>
</msubsup>
</mrow>
<annotation encoding="application/x-tex">
             |AX - B|_F^2
            </annotation>
</semantics>
</math> -->∥ A X − B ∥ F 2 |AX - B|_F^2∥ A X − B ∥ F 2 ​  .
It has shape *(*, k)* .
It is computed when *m > n* and every matrix in `A`  is full-rank,
otherwise, it is an empty tensor.
If `A`  is a batch of matrices and any matrix in the batch is not full rank,
then an empty tensor is returned. This behavior may change in a future PyTorch release.

* *rank* : tensor of ranks of the matrices in `A`  .
It has shape equal to the batch dimensions of `A`  .
It is computed when `driver`  is one of ( *‘gelsy’* , *‘gelsd’* , *‘gelss’* ),
otherwise it is an empty tensor.
* *singular_values* : tensor of singular values of the matrices in `A`  .
It has shape *(*, min(m, n))* .
It is computed when `driver`  is one of ( *‘gelsd’* , *‘gelss’* ),
otherwise it is an empty tensor.

Note 

This function computes *X =*`A` *.pinverse() @*`B`  in a faster and
more numerically stable way than performing the computations separately.

Warning 

The default value of `rcond`  may change in a future PyTorch release.
It is therefore recommended to use a fixed value to avoid potential
breaking changes.

Parameters
:   * **A** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – lhs tensor of shape *(*, m, n)* where *** is zero or more batch dimensions.
* **B** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – rhs tensor of shape *(*, m, k)* where *** is zero or more batch dimensions.
* **rcond** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – used to determine the effective rank of `A`  .
If `rcond` *= None* , `rcond`  is set to the machine
precision of the dtype of `A`  times *max(m, n)* . Default: *None* .

Keyword Arguments
: **driver** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – name of the LAPACK/MAGMA method to be used.
If *None* , *‘gelsy’* is used for CPU inputs and *‘gels’* for CUDA inputs.
Default: *None* .

Returns
:   A named tuple *(solution, residuals, rank, singular_values)* .

Examples: 

```
>>> A = torch.randn(1,3,3)
>>> A
tensor([[[-1.0838,  0.0225,  0.2275],
     [ 0.2438,  0.3844,  0.5499],
     [ 0.1175, -0.9102,  2.0870]]])
>>> B = torch.randn(2,3,3)
>>> B
tensor([[[-0.6772,  0.7758,  0.5109],
     [-1.4382,  1.3769,  1.1818],
     [-0.3450,  0.0806,  0.3967]],
    [[-1.3994, -0.1521, -0.1473],
     [ 1.9194,  1.0458,  0.6705],
     [-1.1802, -0.9796,  1.4086]]])
>>> X = torch.linalg.lstsq(A, B).solution # A is broadcasted to shape (2, 3, 3)
>>> torch.dist(X, torch.linalg.pinv(A) @ B)
tensor(1.5152e-06)

>>> S = torch.linalg.lstsq(A, B, driver='gelsd').singular_values
>>> torch.dist(S, torch.linalg.svdvals(A))
tensor(2.3842e-07)

>>> A[:, 0].zero_()  # Decrease the rank of A
>>> rank = torch.linalg.lstsq(A, B).rank
>>> rank
tensor([2])

```

