torch.addmm 
==========================================================

torch. addmm ( *input*  , *mat1*  , *mat2*  , *out_dtype = None*  , *** , *beta = 1*  , *alpha = 1*  , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Performs a matrix multiplication of the matrices `mat1`  and `mat2`  .
The matrix `input`  is added to the final result. 

If `mat1`  is a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mi>
            n
           </mi>
<mo>
            ×
           </mo>
<mi>
            m
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (n times m)
          </annotation>
</semantics>
</math> -->( n × m ) (n times m)( n × m )  tensor, `mat2`  is a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mi>
            m
           </mi>
<mo>
            ×
           </mo>
<mi>
            p
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (m times p)
          </annotation>
</semantics>
</math> -->( m × p ) (m times p)( m × p )  tensor, then `input`  must be [broadcastable](../notes/broadcasting.html#broadcasting-semantics)  with a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mi>
            n
           </mi>
<mo>
            ×
           </mo>
<mi>
            p
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (n times p)
          </annotation>
</semantics>
</math> -->( n × p ) (n times p)( n × p )  tensor
and `out`  will be a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mi>
            n
           </mi>
<mo>
            ×
           </mo>
<mi>
            p
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (n times p)
          </annotation>
</semantics>
</math> -->( n × p ) (n times p)( n × p )  tensor. 

`alpha`  and `beta`  are scaling factors on matrix-vector product between `mat1`  and `mat2`  and the added matrix `input`  respectively. 

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
            β
           </mi>
<mtext>
            input
           </mtext>
<mo>
            +
           </mo>
<mi>
            α
           </mi>
<mtext>
</mtext>
<mo stretchy="false">
            (
           </mo>
<msub>
<mtext>
             mat1
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo lspace="0.22em" mathvariant="normal" rspace="0.22em">
            @
           </mo>
<msub>
<mtext>
             mat2
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{out} = beta text{input} + alpha (text{mat1}_i mathbin{@} text{mat2}_i)
          </annotation>
</semantics>
</math> -->
out = β input + α ( mat1 i @ mat2 i ) text{out} = beta text{input} + alpha (text{mat1}_i mathbin{@} text{mat2}_i)

out = β input + α ( mat1 i ​ @ mat2 i ​ )

If `beta`  is 0, then the content of `input`  will be ignored, and *nan* and *inf* in
it will not be propagated. 

For inputs of type *FloatTensor* or *DoubleTensor* , arguments `beta`  and `alpha`  must be real numbers, otherwise they should be integers. 

This operation has support for arguments with [sparse layouts](../sparse.html#sparse-docs)  . If `input`  is sparse the result will have the same layout and if `out`  is provided it must have the same layout as `input`  . 

Warning 

Sparse support is a beta feature and some layout(s)/dtype/device combinations may not be supported,
or may not have autograd support. If you notice missing functionality please
open a feature request.

This operator supports [TensorFloat32](../notes/cuda.html#tf32-on-ampere)  . 

On certain ROCm devices, when using float16 inputs this module will use [different precision](../notes/numerical_accuracy.html#fp16-on-mi200)  for backward. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – matrix to be added
* **mat1** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the first matrix to be matrix multiplied
* **mat2** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the second matrix to be matrix multiplied
* **out_dtype** ( [*dtype*](../tensor_attributes.html#torch.dtype "torch.dtype") *,* *optional*  ) – the dtype of the output tensor,
Supported only on CUDA and for torch.float32 given
torch.float16/torch.bfloat16 input dtypes

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

* **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> M = torch.randn(2, 3)
>>> mat1 = torch.randn(2, 3)
>>> mat2 = torch.randn(3, 3)
>>> torch.addmm(M, mat1, mat2)
tensor([[-4.8716,  1.4671, -1.3746],
        [ 0.7573, -3.9555, -2.8681]])

```

