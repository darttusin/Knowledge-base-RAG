torch.matmul 
============================================================

torch. matmul ( *input*  , *other*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Matrix product of two tensors. 

The behavior depends on the dimensionality of the tensors as follows: 

* If both tensors are 1-dimensional, the dot product (scalar) is returned.
* If both arguments are 2-dimensional, the matrix-matrix product is returned.
* If the first argument is 1-dimensional and the second argument is 2-dimensional,
a 1 is prepended to its dimension for the purpose of the matrix multiply.
After the matrix multiply, the prepended dimension is removed.
* If the first argument is 2-dimensional and the second argument is 1-dimensional,
the matrix-vector product is returned.
* If both arguments are at least 1-dimensional and at least one argument is
N-dimensional (where N > 2), then a batched matrix multiply is returned. If the first
argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
batched matrix multiply and removed after. If the second argument is 1-dimensional, a
1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
The non-matrix (i.e. batch) dimensions are [broadcasted](../notes/broadcasting.html#broadcasting-semantics)  (and thus
must be broadcastable). For example, if `input`  is a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mi>
              j
             </mi>
<mo>
              ×
             </mo>
<mn>
              1
             </mn>
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
              n
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             (j times 1 times n times n)
            </annotation>
</semantics>
</math> -->( j × 1 × n × n ) (j times 1 times n times n)( j × 1 × n × n )  tensor and `other`  is a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mi>
              k
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
              n
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             (k times n times n)
            </annotation>
</semantics>
</math> -->( k × n × n ) (k times n times n)( k × n × n )  tensor, `out`  will be a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mi>
              j
             </mi>
<mo>
              ×
             </mo>
<mi>
              k
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
              n
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             (j times k times n times n)
            </annotation>
</semantics>
</math> -->( j × k × n × n ) (j times k times n times n)( j × k × n × n )  tensor.

    Note that the broadcasting logic only looks at the batch dimensions when determining if the inputs
        are broadcastable, and not the matrix dimensions. For example, if `input`  is a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
        <semantics>
        <mrow>
        <mo stretchy="false">
                      (
                     </mo>
        <mi>
                      j
                     </mi>
        <mo>
                      ×
                     </mo>
        <mn>
                      1
                     </mn>
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
                     (j times 1 times n times m)
                    </annotation>
        </semantics>
        </math> -->( j × 1 × n × m ) (j times 1 times n times m)( j × 1 × n × m )  tensor and `other`  is a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
        <semantics>
        <mrow>
        <mo stretchy="false">
                      (
                     </mo>
        <mi>
                      k
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
                     (k times m times p)
                    </annotation>
        </semantics>
        </math> -->( k × m × p ) (k times m times p)( k × m × p )  tensor, these inputs are valid for broadcasting even though the final two dimensions (i.e. the
        matrix dimensions) are different. `out`  will be a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
        <semantics>
        <mrow>
        <mo stretchy="false">
                      (
                     </mo>
        <mi>
                      j
                     </mi>
        <mo>
                      ×
                     </mo>
        <mi>
                      k
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
                     (j times k times n times p)
                    </annotation>
        </semantics>
        </math> -->( j × k × n × p ) (j times k times n times p)( j × k × n × p )  tensor.

This operation has support for arguments with [sparse layouts](../sparse.html#sparse-docs)  . In particular the
matrix-matrix (both arguments 2-dimensional) supports sparse arguments with the same restrictions
as [`torch.mm()`](torch.mm.html#torch.mm "torch.mm") 

Warning 

Sparse support is a beta feature and some layout(s)/dtype/device combinations may not be supported,
or may not have autograd support. If you notice missing functionality please
open a feature request.

This operator supports [TensorFloat32](../notes/cuda.html#tf32-on-ampere)  . 

On certain ROCm devices, when using float16 inputs this module will use [different precision](../notes/numerical_accuracy.html#fp16-on-mi200)  for backward. 

Note 

The 1-dimensional dot product version of this function does not support an `out`  parameter.

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the first tensor to be multiplied
* **other** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the second tensor to be multiplied

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> # vector x vector
>>> tensor1 = torch.randn(3)
>>> tensor2 = torch.randn(3)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([])
>>> # matrix x vector
>>> tensor1 = torch.randn(3, 4)
>>> tensor2 = torch.randn(4)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([3])
>>> # batched matrix x broadcasted vector
>>> tensor1 = torch.randn(10, 3, 4)
>>> tensor2 = torch.randn(4)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([10, 3])
>>> # batched matrix x batched matrix
>>> tensor1 = torch.randn(10, 3, 4)
>>> tensor2 = torch.randn(10, 4, 5)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([10, 3, 5])
>>> # batched matrix x broadcasted matrix
>>> tensor1 = torch.randn(10, 3, 4)
>>> tensor2 = torch.randn(4, 5)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([10, 3, 5])

```

