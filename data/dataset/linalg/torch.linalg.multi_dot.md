torch.linalg.multi_dot 
=================================================================================

torch.linalg. multi_dot ( *tensors*  , *** , *out = None* ) 
:   Efficiently multiplies two or more matrices by reordering the multiplications so that
the fewest arithmetic operations are performed. 

Supports inputs of float, double, cfloat and cdouble dtypes.
This function does not support batched inputs. 

Every tensor in `tensors`  must be 2D, except for the first and last which
may be 1D. If the first tensor is a 1D vector of shape *(n,)* it is treated as a row vector
of shape *(1, n)* , similarly if the last tensor is a 1D vector of shape *(n,)* it is treated
as a column vector of shape *(n, 1)* . 

If the first and last tensors are matrices, the output will be a matrix.
However, if either is a 1D vector, then the output will be a 1D vector. 

Differences with *numpy.linalg.multi_dot* : 

* Unlike *numpy.linalg.multi_dot* , the first and last tensors must either be 1D or 2D
whereas NumPy allows them to be nD

Warning 

This function does not broadcast.

Note 

This function is implemented by chaining [`torch.mm()`](torch.mm.html#torch.mm "torch.mm")  calls after
computing the optimal matrix multiplication order.

Note 

The cost of multiplying two matrices with shapes *(a, b)* and *(b, c)* is *a * b * c* . Given matrices *A* , *B* , *C* with shapes *(10, 100)* , *(100, 5)* , *(5, 50)* respectively, we can calculate the cost of different
multiplication orders as follows: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right left" columnspacing="0em" rowspacing="0.25em">
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mi mathvariant="normal">
                 cost
                </mi>
<mo>
                 ⁡
                </mo>
<mo stretchy="false">
                 (
                </mo>
<mo stretchy="false">
                 (
                </mo>
<mi>
                 A
                </mi>
<mi>
                 B
                </mi>
<mo stretchy="false">
                 )
                </mo>
<mi>
                 C
                </mi>
<mo stretchy="false">
                 )
                </mo>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mo>
                 =
                </mo>
<mn>
                 10
                </mn>
<mo>
                 ×
                </mo>
<mn>
                 100
                </mn>
<mo>
                 ×
                </mo>
<mn>
                 5
                </mn>
<mo>
                 +
                </mo>
<mn>
                 10
                </mn>
<mo>
                 ×
                </mo>
<mn>
                 5
                </mn>
<mo>
                 ×
                </mo>
<mn>
                 50
                </mn>
<mo>
                 =
                </mo>
<mn>
                 7500
                </mn>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mi mathvariant="normal">
                 cost
                </mi>
<mo>
                 ⁡
                </mo>
<mo stretchy="false">
                 (
                </mo>
<mi>
                 A
                </mi>
<mo stretchy="false">
                 (
                </mo>
<mi>
                 B
                </mi>
<mi>
                 C
                </mi>
<mo stretchy="false">
                 )
                </mo>
<mo stretchy="false">
                 )
                </mo>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mo>
                 =
                </mo>
<mn>
                 10
                </mn>
<mo>
                 ×
                </mo>
<mn>
                 100
                </mn>
<mo>
                 ×
                </mo>
<mn>
                 50
                </mn>
<mo>
                 +
                </mo>
<mn>
                 100
                </mn>
<mo>
                 ×
                </mo>
<mn>
                 5
                </mn>
<mo>
                 ×
                </mo>
<mn>
                 50
                </mn>
<mo>
                 =
                </mo>
<mn>
                 75000
                </mn>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
            begin{align*}
operatorname{cost}((AB)C) &amp;= 10 times 100 times 5 + 10 times 5 times 50 = 7500 
operatorname{cost}(A(BC)) &amp;= 10 times 100 times 50 + 100 times 5 times 50 = 75000
end{align*}
           </annotation>
</semantics>
</math> -->
cost ⁡ ( ( A B ) C ) = 10 × 100 × 5 + 10 × 5 × 50 = 7500 cost ⁡ ( A ( B C ) ) = 10 × 100 × 50 + 100 × 5 × 50 = 75000 begin{align*}
operatorname{cost}((AB)C) &= 10 times 100 times 5 + 10 times 5 times 50 = 7500 
operatorname{cost}(A(BC)) &= 10 times 100 times 50 + 100 times 5 times 50 = 75000
end{align*}

cost (( A B ) C ) cost ( A ( BC )) ​ = 10 × 100 × 5 + 10 × 5 × 50 = 7500 = 10 × 100 × 50 + 100 × 5 × 50 = 75000 ​

In this case, multiplying *A* and *B* first followed by *C* is 10 times faster.

Parameters
: **tensors** ( *Sequence* *[* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *]*  ) – two or more tensors to multiply. The first and last
tensors may be 1D or 2D. Every other tensor must be 2D.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – output tensor. Ignored if *None* . Default: *None* .

Examples: 

```
>>> from torch.linalg import multi_dot

>>> multi_dot([torch.tensor([1, 2]), torch.tensor([2, 3])])
tensor(8)
>>> multi_dot([torch.tensor([[1, 2]]), torch.tensor([2, 3])])
tensor([8])
>>> multi_dot([torch.tensor([[1, 2]]), torch.tensor([[2], [3]])])
tensor([[8]])

>>> A = torch.arange(2 * 3).view(2, 3)
>>> B = torch.arange(3 * 2).view(3, 2)
>>> C = torch.arange(2 * 2).view(2, 2)
>>> multi_dot((A, B, C))
tensor([[ 26,  49],
        [ 80, 148]])

```

