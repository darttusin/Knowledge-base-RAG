torch.sparse.sampled_addmm 
=========================================================================================

torch.sparse. sampled_addmm ( *input*  , *mat1*  , *mat2*  , *** , *beta = 1.*  , *alpha = 1.*  , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Performs a matrix multiplication of the dense matrices `mat1`  and `mat2`  at the locations
specified by the sparsity pattern of `input`  . The matrix `input`  is added to the final result. 

Mathematically this performs the following operation: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            out
           </mtext>
<mo>
            =
           </mo>
<mi>
            α
           </mi>
<mtext>
</mtext>
<mo stretchy="false">
            (
           </mo>
<mtext>
            mat1
           </mtext>
<mo lspace="0.22em" mathvariant="normal" rspace="0.22em">
            @
           </mo>
<mtext>
            mat2
           </mtext>
<mo stretchy="false">
            )
           </mo>
<mo>
            ∗
           </mo>
<mtext>
            spy
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mtext>
            input
           </mtext>
<mo stretchy="false">
            )
           </mo>
<mo>
            +
           </mo>
<mi>
            β
           </mi>
<mtext>
            input
           </mtext>
</mrow>
<annotation encoding="application/x-tex">
           text{out} = alpha (text{mat1} mathbin{@} text{mat2})*text{spy}(text{input}) + beta text{input}
          </annotation>
</semantics>
</math> -->
out = α ( mat1 @ mat2 ) ∗ spy ( input ) + β input text{out} = alpha (text{mat1} mathbin{@} text{mat2})*text{spy}(text{input}) + beta text{input}

out = α ( mat1 @ mat2 ) ∗ spy ( input ) + β input

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            spy
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mtext>
            input
           </mtext>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{spy}(text{input})
          </annotation>
</semantics>
</math> -->spy ( input ) text{spy}(text{input})spy ( input )  is the sparsity pattern matrix of `input`  , `alpha`  and `beta`  are the scaling factors. <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            spy
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mtext>
            input
           </mtext>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{spy}(text{input})
          </annotation>
</semantics>
</math> -->spy ( input ) text{spy}(text{input})spy ( input )  has value 1 at the positions where `input`  has non-zero values, and 0 elsewhere. 

Note 

`input`  must be a sparse CSR tensor. `mat1`  and `mat2`  must be dense tensors.

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – a sparse CSR matrix of shape *(m, n)* to be added and used to compute
the sampled matrix multiplication
* **mat1** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – a dense matrix of shape *(m, k)* to be multiplied
* **mat2** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – a dense matrix of shape *(k, n)* to be multiplied

Keyword Arguments
:   * **beta** ( *Number* *,* *optional*  ) – multiplier for `input`  ( <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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

* **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – output tensor. Ignored if *None* . Default: *None* .

Examples: 

```
>>> input = torch.eye(3, device='cuda').to_sparse_csr()
>>> mat1 = torch.randn(3, 5, device='cuda')
>>> mat2 = torch.randn(5, 3, device='cuda')
>>> torch.sparse.sampled_addmm(input, mat1, mat2)
tensor(crow_indices=tensor([0, 1, 2, 3]),
    col_indices=tensor([0, 1, 2]),
    values=tensor([ 0.2847, -0.7805, -0.1900]), device='cuda:0',
    size=(3, 3), nnz=3, layout=torch.sparse_csr)
>>> torch.sparse.sampled_addmm(input, mat1, mat2).to_dense()
tensor([[ 0.2847,  0.0000,  0.0000],
    [ 0.0000, -0.7805,  0.0000],
    [ 0.0000,  0.0000, -0.1900]], device='cuda:0')
>>> torch.sparse.sampled_addmm(input, mat1, mat2, beta=0.5, alpha=0.5)
tensor(crow_indices=tensor([0, 1, 2, 3]),
    col_indices=tensor([0, 1, 2]),
    values=tensor([ 0.1423, -0.3903, -0.0950]), device='cuda:0',
    size=(3, 3), nnz=3, layout=torch.sparse_csr)

```

