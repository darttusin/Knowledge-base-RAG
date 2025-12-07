torch.sspaddmm 
================================================================

torch. sspaddmm ( *input*  , *mat1*  , *mat2*  , *** , *beta = 1*  , *alpha = 1*  , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Matrix multiplies a sparse tensor `mat1`  with a dense tensor `mat2`  , then adds the sparse tensor `input`  to the result. 

Note: This function is equivalent to [`torch.addmm()`](torch.addmm.html#torch.addmm "torch.addmm")  , except `input`  and `mat1`  are sparse. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – a sparse matrix to be added
* **mat1** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – a sparse matrix to be matrix multiplied
* **mat2** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – a dense matrix to be matrix multiplied

Keyword Arguments
:   * **beta** ( *Number* *,* *optional*  ) – multiplier for `mat`  ( <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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

* **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

