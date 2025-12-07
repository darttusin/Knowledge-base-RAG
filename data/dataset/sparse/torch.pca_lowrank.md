torch.pca_lowrank 
=======================================================================

torch. pca_lowrank ( *A*  , *q = None*  , *center = True*  , *niter = 2* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_lowrank.py#L184) 
:   Performs linear Principal Component Analysis (PCA) on a low-rank
matrix, batches of such matrices, or sparse matrix. 

This function returns a namedtuple `(U, S, V)`  which is the
nearly optimal approximation of a singular value decomposition of
a centered matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  such that <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            A
           </mi>
<mo>
            ≈
           </mo>
<mi>
            U
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
<mi>
            S
           </mi>
<mo stretchy="false">
            )
           </mo>
<msup>
<mi>
             V
            </mi>
<mtext>
             H
            </mtext>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           A approx U operatorname{diag}(S) V^{text{H}}
          </annotation>
</semantics>
</math> -->A ≈ U diag ⁡ ( S ) V H A approx U operatorname{diag}(S) V^{text{H}}A ≈ U diag ( S ) V H 

Note 

The relation of `(U, S, V)`  to PCA is as follows: 

* <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  is a data matrix with `m`  samples and `n`  features

* the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->V VV  columns represent the principal directions

* <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
               S
              </mi>
<mo>
               ∗
              </mo>
<mo>
               ∗
              </mo>
<mn>
               2
              </mn>
<mi mathvariant="normal">
               /
              </mi>
<mo stretchy="false">
               (
              </mo>
<mi>
               m
              </mi>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
<mo stretchy="false">
               )
              </mo>
</mrow>
<annotation encoding="application/x-tex">
              S ** 2 / (m - 1)
             </annotation>
</semantics>
</math> -->S ∗ ∗ 2 / ( m − 1 ) S ** 2 / (m - 1)S ∗ ∗ 2/ ( m − 1 )  contains the eigenvalues of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
                A
               </mi>
<mi>
                T
               </mi>
</msup>
<mi>
               A
              </mi>
<mi mathvariant="normal">
               /
              </mi>
<mo stretchy="false">
               (
              </mo>
<mi>
               m
              </mi>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
<mo stretchy="false">
               )
              </mo>
</mrow>
<annotation encoding="application/x-tex">
              A^T A / (m - 1)
             </annotation>
</semantics>
</math> -->A T A / ( m − 1 ) A^T A / (m - 1)A T A / ( m − 1 )  which is the covariance of `A`  when `center=True`  is provided.

* `matmul(A, V[:, :k])`  projects data to the first k
principal components

Note 

Different from the standard SVD, the size of returned
matrices depend on the specified rank and q
values as follows: 

> * <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mi>
> U
> </mi>
> </mrow>
> <annotation encoding="application/x-tex">
> U
> </annotation>
> </semantics>
> </math> -->U UU  is m x q matrix
> 
> * <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mi>
> S
> </mi>
> </mrow>
> <annotation encoding="application/x-tex">
> S
> </annotation>
> </semantics>
> </math> -->S SS  is q-vector
> 
> * <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mi>
> V
> </mi>
> </mrow>
> <annotation encoding="application/x-tex">
> V
> </annotation>
> </semantics>
> </math> -->V VV  is n x q matrix

Note 

To obtain repeatable results, reset the seed for the
pseudorandom number generator

Parameters
:   * **A** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo separator="true">
                ,
               </mo>
<mi>
                m
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                n
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (*, m, n)
              </annotation>
</semantics>
</math> -->( ∗ , m , n ) (*, m, n)( ∗ , m , n )

* **q** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – a slightly overestimated rank of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  . By default, `q = min(6, m, n)`  .

* **center** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – if True, center the input tensor,
otherwise, assume that the input is
centered.
* **niter** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – the number of subspace iterations to
conduct; niter must be a nonnegative
integer, and defaults to 2.

Return type
:   [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [ [torch.Tensor](../tensors.html#torch.Tensor "torch.Tensor")  , [torch.Tensor](../tensors.html#torch.Tensor "torch.Tensor")  , [torch.Tensor](../tensors.html#torch.Tensor "torch.Tensor")  ]

References: 

```
- Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
  structure with randomness: probabilistic algorithms for
  constructing approximate matrix decompositions,
  arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
  `arXiv <http://arxiv.org/abs/0909.4061>`_).

```

