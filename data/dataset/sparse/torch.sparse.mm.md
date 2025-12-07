torch.sparse.mm 
==================================================================

torch.sparse. mm ( ) 
:   > Performs a matrix multiplication of the sparse matrix `mat1`  and the (sparse or strided) matrix `mat2`  . Similar to [`torch.mm()`](torch.mm.html#torch.mm "torch.mm")  , if `mat1`  is a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mo stretchy="false">
> (
> </mo>
> <mi>
> n
> </mi>
> <mo>
> ×
> </mo>
> <mi>
> m
> </mi>
> <mo stretchy="false">
> )
> </mo>
> </mrow>
> <annotation encoding="application/x-tex">
> (n times m)
> </annotation>
> </semantics>
> </math> -->( n × m ) (n times m)( n × m )  tensor, `mat2`  is a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mo stretchy="false">
> (
> </mo>
> <mi>
> m
> </mi>
> <mo>
> ×
> </mo>
> <mi>
> p
> </mi>
> <mo stretchy="false">
> )
> </mo>
> </mrow>
> <annotation encoding="application/x-tex">
> (m times p)
> </annotation>
> </semantics>
> </math> -->( m × p ) (m times p)( m × p )  tensor, out will be a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
> <semantics>
> <mrow>
> <mo stretchy="false">
> (
> </mo>
> <mi>
> n
> </mi>
> <mo>
> ×
> </mo>
> <mi>
> p
> </mi>
> <mo stretchy="false">
> )
> </mo>
> </mrow>
> <annotation encoding="application/x-tex">
> (n times p)
> </annotation>
> </semantics>
> </math> -->( n × p ) (n times p)( n × p )  tensor.
> When `mat1`  is a COO tensor it must have *sparse_dim = 2* .
> When inputs are COO tensors, this function also supports backward for both inputs. 
> 
> Supports both CSR and COO storage formats.

Note 

This function doesn’t support computing derivatives with respect to CSR matrices. 

This function also additionally accepts an optional `reduce`  argument that allows
specification of an optional reduction operation, mathematically performs the following operation:

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             z
            </mi>
<mrow>
<mi>
              i
             </mi>
<mi>
              j
             </mi>
</mrow>
</msub>
<mo>
            =
           </mo>
<munderover>
<mo>
             ⨁
            </mo>
<mrow>
<mi>
              k
             </mi>
<mo>
              =
             </mo>
<mn>
              0
             </mn>
</mrow>
<mrow>
<mi>
              K
             </mi>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
</mrow>
</munderover>
<msub>
<mi>
             x
            </mi>
<mrow>
<mi>
              i
             </mi>
<mi>
              k
             </mi>
</mrow>
</msub>
<msub>
<mi>
             y
            </mi>
<mrow>
<mi>
              k
             </mi>
<mi>
              j
             </mi>
</mrow>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           z_{ij} = bigoplus_{k = 0}^{K - 1} x_{ik} y_{kj}
          </annotation>
</semantics>
</math> -->
z i j = ⨁ k = 0 K − 1 x i k y k j z_{ij} = bigoplus_{k = 0}^{K - 1} x_{ik} y_{kj}

z ij ​ = k = 0 ⨁ K − 1 ​ x ik ​ y kj ​

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
            ⨁
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           bigoplus
          </annotation>
</semantics>
</math> -->⨁ bigoplus⨁  defines the reduce operator. `reduce`  is implemented only for
CSR storage format on CPU device. 

Parameters
:   * **mat1** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the first sparse matrix to be multiplied
* **mat2** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the second matrix to be multiplied, which could be sparse or dense
* **reduce** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – the reduction operation to apply for non-unique indices
( `"sum"`  , `"mean"`  , `"amax"`  , `"amin"`  ). Default `"sum"`  .

Shape:
:   The format of the output tensor of this function follows:
- sparse x sparse -> sparse
- sparse x dense -> dense

Example: 

```
>>> a = torch.tensor([[1., 0, 2], [0, 3, 0]]).to_sparse().requires_grad_()
>>> a
tensor(indices=tensor([[0, 0, 1],
                       [0, 2, 1]]),
       values=tensor([1., 2., 3.]),
       size=(2, 3), nnz=3, layout=torch.sparse_coo, requires_grad=True)
>>> b = torch.tensor([[0, 1.], [2, 0], [0, 0]], requires_grad=True)
>>> b
tensor([[0., 1.],
        [2., 0.],
        [0., 0.]], requires_grad=True)
>>> y = torch.sparse.mm(a, b)
>>> y
tensor([[0., 1.],
        [6., 0.]], grad_fn=<SparseAddmmBackward0>)
>>> y.sum().backward()
>>> a.grad
tensor(indices=tensor([[0, 0, 1],
                       [0, 2, 1]]),
       values=tensor([1., 0., 2.]),
       size=(2, 3), nnz=3, layout=torch.sparse_coo)
>>> c = a.detach().to_sparse_csr()
>>> c
tensor(crow_indices=tensor([0, 2, 3]),
       col_indices=tensor([0, 2, 1]),
       values=tensor([1., 2., 3.]), size=(2, 3), nnz=3,
       layout=torch.sparse_csr)
>>> y1 = torch.sparse.mm(c, b, 'sum')
>>> y1
tensor([[0., 1.],
        [6., 0.]], grad_fn=<SparseMmReduceImplBackward0>)
>>> y2 = torch.sparse.mm(c, b, 'max')
>>> y2
tensor([[0., 1.],
        [6., 0.]], grad_fn=<SparseMmReduceImplBackward0>)

```

