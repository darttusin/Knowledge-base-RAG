torch.linalg.svd 
====================================================================

torch.linalg. svd ( *A*  , *full_matrices = True*  , *** , *driver = None*  , *out = None* ) 
:   Computes the singular value decomposition (SVD) of a matrix. 

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
the **full SVD** of a matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A ∈ K m × n A in mathbb{K}^{m times n}A ∈ K m × n  , if *k = min(m,n)* , is defined as 

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
<mpadded width="0px">
<mrow>
<mspace width="2em">
</mspace>
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
              S
             </mi>
<mo>
              ∈
             </mo>
<msup>
<mi mathvariant="double-struck">
               R
              </mi>
<mi>
               k
              </mi>
</msup>
<mo separator="true">
              ,
             </mo>
<mi>
              V
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
</mpadded>
</mrow>
<annotation encoding="application/x-tex">
           A = U operatorname{diag}(S) V^{text{H}}
mathrlap{qquad U in mathbb{K}^{m times m}, S in mathbb{R}^k, V in mathbb{K}^{n times n}}
          </annotation>
</semantics>
</math> -->
A = U diag ⁡ ( S ) V H U ∈ K m × m , S ∈ R k , V ∈ K n × n A = U operatorname{diag}(S) V^{text{H}}
mathrlap{qquad U in mathbb{K}^{m times m}, S in mathbb{R}^k, V in mathbb{K}^{n times n}}

A = U diag ( S ) V H U ∈ K m × m , S ∈ R k , V ∈ K n × n

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
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
           operatorname{diag}(S) in mathbb{K}^{m times n}
          </annotation>
</semantics>
</math> -->diag ⁡ ( S ) ∈ K m × n operatorname{diag}(S) in mathbb{K}^{m times n}diag ( S ) ∈ K m × n  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
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
           V^{text{H}}
          </annotation>
</semantics>
</math> -->V H V^{text{H}}V H  is the conjugate transpose when <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->V VV  is complex, and the transpose when <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->V VV  is real-valued.
The matrices <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->U UU  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->V VV  (and thus <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
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
           V^{text{H}}
          </annotation>
</semantics>
</math> -->V H V^{text{H}}V H  ) are orthogonal in the real case, and unitary in the complex case. 

When *m > n* (resp. *m < n* ) we can drop the last *m - n* (resp. *n - m* ) columns of *U* (resp. *V* ) to form the **reduced SVD** : 

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
<mpadded width="0px">
<mrow>
<mspace width="2em">
</mspace>
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
              S
             </mi>
<mo>
              ∈
             </mo>
<msup>
<mi mathvariant="double-struck">
               R
              </mi>
<mi>
               k
              </mi>
</msup>
<mo separator="true">
              ,
             </mo>
<mi>
              V
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
</mpadded>
</mrow>
<annotation encoding="application/x-tex">
           A = U operatorname{diag}(S) V^{text{H}}
mathrlap{qquad U in mathbb{K}^{m times k}, S in mathbb{R}^k, V in mathbb{K}^{n times k}}
          </annotation>
</semantics>
</math> -->
A = U diag ⁡ ( S ) V H U ∈ K m × k , S ∈ R k , V ∈ K n × k A = U operatorname{diag}(S) V^{text{H}}
mathrlap{qquad U in mathbb{K}^{m times k}, S in mathbb{R}^k, V in mathbb{K}^{n times k}}

A = U diag ( S ) V H U ∈ K m × k , S ∈ R k , V ∈ K n × k

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
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
              k
             </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           operatorname{diag}(S) in mathbb{K}^{k times k}
          </annotation>
</semantics>
</math> -->diag ⁡ ( S ) ∈ K k × k operatorname{diag}(S) in mathbb{K}^{k times k}diag ( S ) ∈ K k × k  .
In this case, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->U UU  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->V VV  also have orthonormal columns. 

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if `A`  is a batch of matrices then
the output has the same batch dimensions. 

The returned decomposition is a named tuple *(U, S, Vh)* which corresponds to <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->U UU  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            S
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           S
          </annotation>
</semantics>
</math> -->S SS  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
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
           V^{text{H}}
          </annotation>
</semantics>
</math> -->V H V^{text{H}}V H  above. 

The singular values are returned in descending order. 

The parameter `full_matrices`  chooses between the full (default) and reduced SVD. 

The `driver`  kwarg may be used in CUDA with a cuSOLVER backend to choose the algorithm used to compute the SVD.
The choice of a driver is a trade-off between accuracy and speed. 

* If `A`  is well-conditioned (its [condition number](https://localhost:8000/docs/main/linalg.html#torch.linalg.cond)  is not too large), or you do not mind some precision loss.

    + For a general matrix: *‘gesvdj’* (Jacobi method)
        + If `A`  is tall or wide ( *m >> n* or *m << n* ): *‘gesvda’* (Approximate method)
* If `A`  is not well-conditioned or precision is relevant: *‘gesvd’* (QR based)

By default ( `driver` *= None* ), we call *‘gesvdj’* and, if it fails, we fallback to *‘gesvd’* . 

Differences with *numpy.linalg.svd* : 

* Unlike *numpy.linalg.svd* , this function always returns a tuple of three tensors
and it doesn’t support *compute_uv* argument.
Please use [`torch.linalg.svdvals()`](torch.linalg.svdvals.html#torch.linalg.svdvals "torch.linalg.svdvals")  , which computes only the singular values,
instead of *compute_uv=False* .

Note 

When `full_matrices` *= True* , the gradients with respect to *U[…, :, min(m, n):]* and *Vh[…, min(m, n):, :]* will be ignored, as those vectors can be arbitrary bases
of the corresponding subspaces.

Warning 

The returned tensors *U* and *V* are not unique, nor are they continuous with
respect to `A`  .
Due to this lack of uniqueness, different hardware and software may compute
different singular vectors. 

This non-uniqueness is caused by the fact that multiplying any pair of singular
vectors <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              u
             </mi>
<mi>
              k
             </mi>
</msub>
<mo separator="true">
             ,
            </mo>
<msub>
<mi>
              v
             </mi>
<mi>
              k
             </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            u_k, v_k
           </annotation>
</semantics>
</math> -->u k , v k u_k, v_ku k ​ , v k ​  by *-1* in the real case or by <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->e i ϕ , ϕ ∈ R e^{i phi}, phi in mathbb{R}e i ϕ , ϕ ∈ R  in the complex case produces another two
valid singular vectors of the matrix.
For this reason, the loss function shall not depend on this <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</mrow>
<annotation encoding="application/x-tex">
            e^{i phi}
           </annotation>
</semantics>
</math> -->e i ϕ e^{i phi}e i ϕ  quantity,
as it is not well-defined.
This is checked for complex inputs when computing the gradients of this function. As such,
when inputs are complex and are on a CUDA device, the computation of the gradients
of this function synchronizes that device with the CPU.

Warning 

Gradients computed using *U* or *Vh* will only be finite when `A`  does not have repeated singular values. If `A`  is rectangular,
additionally, zero must also not be one of its singular values.
Furthermore, if the distance between any two singular values is close to zero,
the gradient will be numerically unstable, as it depends on the singular values <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->σ i sigma_iσ i ​  through the computation of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<msubsup>
<mi>
                σ
               </mi>
<mi>
                i
               </mi>
<mn>
                2
               </mn>
</msubsup>
<mo>
               −
              </mo>
<msubsup>
<mi>
                σ
               </mi>
<mi>
                j
               </mi>
<mn>
                2
               </mn>
</msubsup>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
            frac{1}{min_{i neq j} sigma_i^2 - sigma_j^2}
           </annotation>
</semantics>
</math> -->1 min ⁡ i ≠ j σ i 2 − σ j 2 frac{1}{min_{i neq j} sigma_i^2 - sigma_j^2}m i n i  = j ​ σ i 2 ​ − σ j 2 ​ 1 ​  .
In the rectangular case, the gradient will also be numerically unstable when `A`  has small singular values, as it also depends on the computation of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mfrac>
<mn>
              1
             </mn>
<msub>
<mi>
               σ
              </mi>
<mi>
               i
              </mi>
</msub>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
            frac{1}{sigma_i}
           </annotation>
</semantics>
</math> -->1 σ i frac{1}{sigma_i}σ i ​ 1 ​  .

See also 

[`torch.linalg.svdvals()`](torch.linalg.svdvals.html#torch.linalg.svdvals "torch.linalg.svdvals")  computes only the singular values.
Unlike [`torch.linalg.svd()`](#torch.linalg.svd "torch.linalg.svd")  , the gradients of [`svdvals()`](torch.linalg.svdvals.html#torch.linalg.svdvals "torch.linalg.svdvals")  are always
numerically stable. 

[`torch.linalg.eig()`](torch.linalg.eig.html#torch.linalg.eig "torch.linalg.eig")  for a function that computes another type of spectral
decomposition of a matrix. The eigendecomposition works just on square matrices. 

[`torch.linalg.eigh()`](torch.linalg.eigh.html#torch.linalg.eigh "torch.linalg.eigh")  for a (faster) function that computes the eigenvalue decomposition
for Hermitian and symmetric matrices. 

[`torch.linalg.qr()`](torch.linalg.qr.html#torch.linalg.qr "torch.linalg.qr")  for another (much faster) decomposition that works on general
matrices.

Parameters
:   * **A** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, m, n)* where *** is zero or more batch dimensions.
* **full_matrices** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – controls whether to compute the full or reduced
SVD, and consequently,
the shape of the returned tensors *U* and *Vh* . Default: *True* .

Keyword Arguments
:   * **driver** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – name of the cuSOLVER method to be used. This keyword argument only works on CUDA inputs.
Available options are: *None* , *gesvd* , *gesvdj* , and *gesvda* .
Default: *None* .
* **out** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – output tuple of three tensors. Ignored if *None* .

Returns
:   A named tuple *(U, S, Vh)* which corresponds to <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->U UU  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
              S
             </mi>
</mrow>
<annotation encoding="application/x-tex">
             S
            </annotation>
</semantics>
</math> -->S SS  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
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
             V^{text{H}}
            </annotation>
</semantics>
</math> -->V H V^{text{H}}V H  above. 

*S* will always be real-valued, even when `A`  is complex.
It will also be ordered in descending order. 

*U* and *Vh* will have the same dtype as `A`  . The left / right singular vectors will be given by
the columns of *U* and the rows of *Vh* respectively.

Examples: 

```
>>> A = torch.randn(5, 3)
>>> U, S, Vh = torch.linalg.svd(A, full_matrices=False)
>>> U.shape, S.shape, Vh.shape
(torch.Size([5, 3]), torch.Size([3]), torch.Size([3, 3]))
>>> torch.dist(A, U @ torch.diag(S) @ Vh)
tensor(1.0486e-06)

>>> U, S, Vh = torch.linalg.svd(A)
>>> U.shape, S.shape, Vh.shape
(torch.Size([5, 5]), torch.Size([3]), torch.Size([3, 3]))
>>> torch.dist(A, U[:, :3] @ torch.diag(S) @ Vh)
tensor(1.0486e-06)

>>> A = torch.randn(7, 5, 3)
>>> U, S, Vh = torch.linalg.svd(A, full_matrices=False)
>>> torch.dist(A, U @ torch.diag_embed(S) @ Vh)
tensor(3.0957e-06)

```

