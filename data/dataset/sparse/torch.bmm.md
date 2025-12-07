torch.bmm 
======================================================

torch. bmm ( *input*  , *mat2*  , *out_dtype = None*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Performs a batch matrix-matrix product of matrices stored in `input`  and `mat2`  . 

`input`  and `mat2`  must be 3-D tensors each containing
the same number of matrices. 

If `input`  is a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mi>
            b
           </mi>
<mo>
            ×
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
           (b times n times m)
          </annotation>
</semantics>
</math> -->( b × n × m ) (b times n times m)( b × n × m )  tensor, `mat2`  is a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mi>
            b
           </mi>
<mo>
            ×
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
           (b times m times p)
          </annotation>
</semantics>
</math> -->( b × m × p ) (b times m times p)( b × m × p )  tensor, `out`  will be a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mi>
            b
           </mi>
<mo>
            ×
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
           (b times n times p)
          </annotation>
</semantics>
</math> -->( b × n × p ) (b times n times p)( b × n × p )  tensor. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mtext>
             out
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo>
            =
           </mo>
<msub>
<mtext>
             input
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
</mrow>
<annotation encoding="application/x-tex">
           text{out}_i = text{input}_i mathbin{@} text{mat2}_i
          </annotation>
</semantics>
</math> -->
out i = input i @ mat2 i text{out}_i = text{input}_i mathbin{@} text{mat2}_i

out i ​ = input i ​ @ mat2 i ​

This operator supports [TensorFloat32](../notes/cuda.html#tf32-on-ampere)  . 

On certain ROCm devices, when using float16 inputs this module will use [different precision](../notes/numerical_accuracy.html#fp16-on-mi200)  for backward. 

Note 

This function does not [broadcast](../notes/broadcasting.html#broadcasting-semantics)  .
For broadcasting matrix products, see [`torch.matmul()`](torch.matmul.html#torch.matmul "torch.matmul")  .

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the first batch of matrices to be multiplied
* **mat2** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the second batch of matrices to be multiplied
* **out_dtype** ( [*dtype*](../tensor_attributes.html#torch.dtype "torch.dtype") *,* *optional*  ) – the dtype of the output tensor,
Supported only on CUDA and for torch.float32 given
torch.float16/torch.bfloat16 input dtypes

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> input = torch.randn(10, 3, 4)
>>> mat2 = torch.randn(10, 4, 5)
>>> res = torch.bmm(input, mat2)
>>> res.size()
torch.Size([10, 3, 5])

```

