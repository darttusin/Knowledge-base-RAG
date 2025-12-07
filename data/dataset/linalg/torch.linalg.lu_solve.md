torch.linalg.lu_solve 
===============================================================================

torch.linalg. lu_solve ( *LU*  , *pivots*  , *B*  , *** , *left = True*  , *adjoint = False*  , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the solution of a square system of linear equations with a unique solution given an LU decomposition. 

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
this function computes the solution <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
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
<annotation encoding="application/x-tex">
           X in mathbb{K}^{n times k}
          </annotation>
</semantics>
</math> -->X ∈ K n × k X in mathbb{K}^{n times k}X ∈ K n × k  of the **linear system** associated to <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<annotation encoding="application/x-tex">
           A in mathbb{K}^{n times n}, B in mathbb{K}^{n times k}
          </annotation>
</semantics>
</math> -->A ∈ K n × n , B ∈ K n × k A in mathbb{K}^{n times n}, B in mathbb{K}^{n times k}A ∈ K n × n , B ∈ K n × k  , which is defined as 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->
A X = B AX = B

A X = B

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  is given factorized as returned by [`lu_factor()`](torch.linalg.lu_factor.html#torch.linalg.lu_factor "torch.linalg.lu_factor")  . 

If `left` *= False* , this function returns the matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
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
<annotation encoding="application/x-tex">
           X in mathbb{K}^{n times k}
          </annotation>
</semantics>
</math> -->X ∈ K n × k X in mathbb{K}^{n times k}X ∈ K n × k  that solves the system 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            X
           </mi>
<mi>
            A
           </mi>
<mo>
            =
           </mo>
<mi>
            B
           </mi>
<mpadded width="0px">
<mrow>
<mspace width="2em">
</mspace>
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
<mi mathvariant="normal">
              .
             </mi>
</mrow>
</mpadded>
</mrow>
<annotation encoding="application/x-tex">
           XA = Bmathrlap{qquad A in mathbb{K}^{k times k}, B in mathbb{K}^{n times k}.}
          </annotation>
</semantics>
</math> -->
X A = B A ∈ K k × k , B ∈ K n × k . XA = Bmathrlap{qquad A in mathbb{K}^{k times k}, B in mathbb{K}^{n times k}.}

X A = B A ∈ K k × k , B ∈ K n × k .

If `adjoint` *= True* (and `left` *= True* ), given an LU factorization of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  this function function returns the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
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
<annotation encoding="application/x-tex">
           X in mathbb{K}^{n times k}
          </annotation>
</semantics>
</math> -->X ∈ K n × k X in mathbb{K}^{n times k}X ∈ K n × k  that solves the system 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
             A
            </mi>
<mtext>
             H
            </mtext>
</msup>
<mi>
            X
           </mi>
<mo>
            =
           </mo>
<mi>
            B
           </mi>
<mpadded width="0px">
<mrow>
<mspace width="2em">
</mspace>
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
<mi mathvariant="normal">
              .
             </mi>
</mrow>
</mpadded>
</mrow>
<annotation encoding="application/x-tex">
           A^{text{H}}X = Bmathrlap{qquad A in mathbb{K}^{k times k}, B in mathbb{K}^{n times k}.}
          </annotation>
</semantics>
</math> -->
A H X = B A ∈ K k × k , B ∈ K n × k . A^{text{H}}X = Bmathrlap{qquad A in mathbb{K}^{k times k}, B in mathbb{K}^{n times k}.}

A H X = B A ∈ K k × k , B ∈ K n × k .

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
             A
            </mi>
<mtext>
             H
            </mtext>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           A^{text{H}}
          </annotation>
</semantics>
</math> -->A H A^{text{H}}A H  is the conjugate transpose when <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  is complex, and the
transpose when <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  is real-valued. The `left` *= False* case is analogous. 

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if the inputs are batches of matrices then
the output has the same batch dimensions. 

Parameters
:   * **LU** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, n, n)* (or *(*, k, k)* if `left` *= True* )
where *** is zero or more batch dimensions as returned by [`lu_factor()`](torch.linalg.lu_factor.html#torch.linalg.lu_factor "torch.linalg.lu_factor")  .
* **pivots** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, n)* (or *(*, k)* if `left` *= True* )
where *** is zero or more batch dimensions as returned by [`lu_factor()`](torch.linalg.lu_factor.html#torch.linalg.lu_factor "torch.linalg.lu_factor")  .
* **B** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – right-hand side tensor of shape *(*, n, k)* .

Keyword Arguments
:   * **left** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether to solve the system <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
               AX=B
              </annotation>
</semantics>
</math> -->A X = B AX=BA X = B  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                X
               </mi>
<mi>
                A
               </mi>
<mo>
                =
               </mo>
<mi>
                B
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               XA = B
              </annotation>
</semantics>
</math> -->X A = B XA = BX A = B  . Default: *True* .

* **adjoint** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether to solve the system <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
               AX=B
              </annotation>
</semantics>
</math> -->A X = B AX=BA X = B  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
                 A
                </mi>
<mtext>
                 H
                </mtext>
</msup>
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
               A^{text{H}}X = B
              </annotation>
</semantics>
</math> -->A H X = B A^{text{H}}X = BA H X = B  . Default: *False* .

* **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – output tensor. Ignored if *None* . Default: *None* .

Examples: 

```
>>> A = torch.randn(3, 3)
>>> LU, pivots = torch.linalg.lu_factor(A)
>>> B = torch.randn(3, 2)
>>> X = torch.linalg.lu_solve(LU, pivots, B)
>>> torch.allclose(A @ X, B)
True

>>> B = torch.randn(3, 3, 2)   # Broadcasting rules apply: A is broadcasted
>>> X = torch.linalg.lu_solve(LU, pivots, B)
>>> torch.allclose(A @ X, B)
True

>>> B = torch.randn(3, 5, 3)
>>> X = torch.linalg.lu_solve(LU, pivots, B, left=False)
>>> torch.allclose(X @ A, B)
True

>>> B = torch.randn(3, 3, 4)   # Now solve for A^T
>>> X = torch.linalg.lu_solve(LU, pivots, B, adjoint=True)
>>> torch.allclose(A.mT @ X, B)
True

```

