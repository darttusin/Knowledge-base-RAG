torch.nonzero 
==============================================================

torch. nonzero ( *input*  , *** , *out = None*  , *as_tuple = False* ) → LongTensor or tuple of LongTensors 
:   Note 

[`torch.nonzero(..., as_tuple=False)`](#torch.nonzero "torch.nonzero")  (default) returns a
2-D tensor where each row is the index for a nonzero value. 

[`torch.nonzero(..., as_tuple=True)`](#torch.nonzero "torch.nonzero")  returns a tuple of 1-D
index tensors, allowing for advanced indexing, so `x[x.nonzero(as_tuple=True)]`  gives all nonzero values of tensor `x`  . Of the returned tuple, each index tensor
contains nonzero indices for a certain dimension. 

See below for more details on the two behaviors. 

When `input`  is on CUDA, [`torch.nonzero()`](#torch.nonzero "torch.nonzero")  causes
host-device synchronization.

**When** `as_tuple` **is** `False` **(default)** : 

Returns a tensor containing the indices of all non-zero elements of `input`  . Each row in the result contains the indices of a non-zero
element in `input`  . The result is sorted lexicographically, with
the last index changing the fastest (C-style). 

If `input`  has <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            n
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           n
          </annotation>
</semantics>
</math> -->n nn  dimensions, then the resulting indices tensor `out`  is of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mi>
            z
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
           (z times n)
          </annotation>
</semantics>
</math> -->( z × n ) (z times n)( z × n )  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            z
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           z
          </annotation>
</semantics>
</math> -->z zz  is the total number of
non-zero elements in the `input`  tensor. 

**When** `as_tuple` **is** `True`  : 

Returns a tuple of 1-D tensors, one for each dimension in `input`  ,
each containing the indices (in that dimension) of all non-zero elements of `input`  . 

If `input`  has <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            n
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           n
          </annotation>
</semantics>
</math> -->n nn  dimensions, then the resulting tuple contains <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            n
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           n
          </annotation>
</semantics>
</math> -->n nn  tensors of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            z
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           z
          </annotation>
</semantics>
</math> -->z zz  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            z
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           z
          </annotation>
</semantics>
</math> -->z zz  is the total number of
non-zero elements in the `input`  tensor. 

As a special case, when `input`  has zero dimensions and a nonzero scalar
value, it is treated as a one-dimensional tensor with one element. 

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( *LongTensor* *,* *optional*  ) – the output tensor containing indices

Returns
:   If `as_tuple`  is `False`  , the output
tensor containing indices. If `as_tuple`  is `True`  , one 1-D tensor for
each dimension, containing the indices of each nonzero element along that
dimension.

Return type
:   LongTensor or [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  of LongTensor

Example: 

```
>>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]))
tensor([[ 0],
        [ 1],
        [ 2],
        [ 4]])
>>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
...                             [0.0, 0.4, 0.0, 0.0],
...                             [0.0, 0.0, 1.2, 0.0],
...                             [0.0, 0.0, 0.0,-0.4]]))
tensor([[ 0,  0],
        [ 1,  1],
        [ 2,  2],
        [ 3,  3]])
>>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]), as_tuple=True)
(tensor([0, 1, 2, 4]),)
>>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
...                             [0.0, 0.4, 0.0, 0.0],
...                             [0.0, 0.0, 1.2, 0.0],
...                             [0.0, 0.0, 0.0,-0.4]]), as_tuple=True)
(tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3]))
>>> torch.nonzero(torch.tensor(5), as_tuple=True)
(tensor([0]),)

```

