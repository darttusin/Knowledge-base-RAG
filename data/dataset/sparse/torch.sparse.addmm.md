torch.sparse.addmm 
========================================================================

torch.sparse. addmm ( *mat*  , *mat1*  , *mat2*  , *** , *beta = 1.*  , *alpha = 1.* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   This function does exact same thing as [`torch.addmm()`](torch.addmm.html#torch.addmm "torch.addmm")  in the forward,
except that it supports backward for sparse COO matrix `mat1`  .
When `mat1`  is a COO tensor it must have *sparse_dim = 2* .
When inputs are COO tensors, this function also supports backward for both inputs. 

Supports both CSR and COO storage formats. 

Note 

This function doesn’t support computing derivatives with respect to CSR matrices.

Parameters
:   * **mat** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – a dense matrix to be added
* **mat1** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – a sparse matrix to be multiplied
* **mat2** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – a dense matrix to be multiplied
* **beta** ( *Number* *,* *optional*  ) – multiplier for `mat`  ( <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                β
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               beta
              </annotation>
</semantics>
</math> -->β betaβ  )

* **alpha** ( *Number* *,* *optional*  ) – multiplier for <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                m
               </mi>
<mi>
                a
               </mi>
<mi>
                t
               </mi>
<mn>
                1
               </mn>
<mi mathvariant="normal">
                @
               </mi>
<mi>
                m
               </mi>
<mi>
                a
               </mi>
<mi>
                t
               </mi>
<mn>
                2
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               mat1 @ mat2
              </annotation>
</semantics>
</math> -->m a t 1 @ m a t 2 mat1 @ mat2ma t 1@ ma t 2  ( <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                α
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               alpha
              </annotation>
</semantics>
</math> -->α alphaα  )

