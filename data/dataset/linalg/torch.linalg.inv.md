torch.linalg.inv 
====================================================================

torch.linalg. inv ( *A*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the inverse of a square matrix if it exists.
Throws a *RuntimeError* if the matrix is not invertible. 

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
for a matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</mrow>
<annotation encoding="application/x-tex">
           A in mathbb{K}^{n times n}
          </annotation>
</semantics>
</math> -->A ∈ K n × n A in mathbb{K}^{n times n}A ∈ K n × n  ,
its **inverse matrix** <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
             A
            </mi>
<mrow>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
</mrow>
</msup>
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
</mrow>
<annotation encoding="application/x-tex">
           A^{-1} in mathbb{K}^{n times n}
          </annotation>
</semantics>
</math> -->A − 1 ∈ K n × n A^{-1} in mathbb{K}^{n times n}A − 1 ∈ K n × n  (if it exists) is defined as 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
             A
            </mi>
<mrow>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
</mrow>
</msup>
<mi>
            A
           </mi>
<mo>
            =
           </mo>
<mi>
            A
           </mi>
<msup>
<mi>
             A
            </mi>
<mrow>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
</mrow>
</msup>
<mo>
            =
           </mo>
<msub>
<mi mathvariant="normal">
             I
            </mi>
<mi>
             n
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           A^{-1}A = AA^{-1} = mathrm{I}_n
          </annotation>
</semantics>
</math> -->
A − 1 A = A A − 1 = I n A^{-1}A = AA^{-1} = mathrm{I}_n

A − 1 A = A A − 1 = I n ​

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi mathvariant="normal">
             I
            </mi>
<mi>
             n
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           mathrm{I}_n
          </annotation>
</semantics>
</math> -->I n mathrm{I}_nI n ​  is the *n* -dimensional identity matrix. 

The inverse matrix exists if and only if <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  is [invertible](https://en.wikipedia.org/wiki/Invertible_matrix#The_invertible_matrix_theorem)  . In this case,
the inverse is unique. 

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if `A`  is a batch of matrices
then the output has the same batch dimensions. 

Note 

When inputs are on a CUDA device, this function synchronizes that device with the CPU. For a version of this function that does not synchronize, see [`torch.linalg.inv_ex()`](torch.linalg.inv_ex.html#torch.linalg.inv_ex "torch.linalg.inv_ex")  .

Note 

Consider using [`torch.linalg.solve()`](torch.linalg.solve.html#torch.linalg.solve "torch.linalg.solve")  if possible for multiplying a matrix on the left by
the inverse, as: 

```
linalg.solve(A, B) == linalg.inv(A) @ B  # When B is a matrix

```

It is always preferred to use [`solve()`](torch.linalg.solve.html#torch.linalg.solve "torch.linalg.solve")  when possible, as it is faster and more
numerically stable than computing the inverse explicitly.

See also 

[`torch.linalg.pinv()`](torch.linalg.pinv.html#torch.linalg.pinv "torch.linalg.pinv")  computes the pseudoinverse (Moore-Penrose inverse) of matrices
of any shape. 

[`torch.linalg.solve()`](torch.linalg.solve.html#torch.linalg.solve "torch.linalg.solve")  computes `A` *.inv() @*`B`  with a
numerically stable algorithm.

Parameters
: **A** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, n, n)* where *** is zero or more batch dimensions
consisting of invertible matrices.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – output tensor. Ignored if *None* . Default: *None* .

Raises
:   [**RuntimeError**](https://docs.python.org/3/library/exceptions.html#RuntimeError "(in Python v3.13)")  – if the matrix `A`  or any matrix in the batch of matrices `A`  is not invertible.

Examples: 

```
>>> A = torch.randn(4, 4)
>>> Ainv = torch.linalg.inv(A)
>>> torch.dist(A @ Ainv, torch.eye(4))
tensor(1.1921e-07)

>>> A = torch.randn(2, 3, 4, 4)  # Batch of matrices
>>> Ainv = torch.linalg.inv(A)
>>> torch.dist(A @ Ainv, torch.eye(4))
tensor(1.9073e-06)

>>> A = torch.randn(4, 4, dtype=torch.complex128)  # Complex matrix
>>> Ainv = torch.linalg.inv(A)
>>> torch.dist(A @ Ainv, torch.eye(4))
tensor(7.5107e-16, dtype=torch.float64)

```

