torch.linalg.solve 
========================================================================

torch.linalg. solve ( *A*  , *B*  , *** , *left = True*  , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the solution of a square system of linear equations with a unique solution. 

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
this function computes the solution <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            X
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
              k
             </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           X in mathbb{K}^{n times k}
          </annotation>
</semantics>
</math> -->X ∈ K n × k X in mathbb{K}^{n times k}X ∈ K n × k  of the **linear system** associated to <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mo separator="true">
            ,
           </mo>
<mi>
            B
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
              k
             </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           A in mathbb{K}^{n times n}, B in mathbb{K}^{n times k}
          </annotation>
</semantics>
</math> -->A ∈ K n × n , B ∈ K n × k A in mathbb{K}^{n times n}, B in mathbb{K}^{n times k}A ∈ K n × n , B ∈ K n × k  , which is defined as 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            A
           </mi>
<mi>
            X
           </mi>
<mo>
            =
           </mo>
<mi>
            B
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           AX = B
          </annotation>
</semantics>
</math> -->
A X = B AX = B

A X = B

If `left` *= False* , this function returns the matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            X
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
              k
             </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           X in mathbb{K}^{n times k}
          </annotation>
</semantics>
</math> -->X ∈ K n × k X in mathbb{K}^{n times k}X ∈ K n × k  that solves the system 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            X
           </mi>
<mi>
            A
           </mi>
<mo>
            =
           </mo>
<mi>
            B
           </mi>
<mpadded width="0px">
<mrow>
<mspace width="2em">
</mspace>
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
                k
               </mi>
<mo>
                ×
               </mo>
<mi>
                k
               </mi>
</mrow>
</msup>
<mo separator="true">
              ,
             </mo>
<mi>
              B
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
                k
               </mi>
</mrow>
</msup>
<mi mathvariant="normal">
              .
             </mi>
</mrow>
</mpadded>
</mrow>
<annotation encoding="application/x-tex">
           XA = Bmathrlap{qquad A in mathbb{K}^{k times k}, B in mathbb{K}^{n times k}.}
          </annotation>
</semantics>
</math> -->
X A = B A ∈ K k × k , B ∈ K n × k . XA = Bmathrlap{qquad A in mathbb{K}^{k times k}, B in mathbb{K}^{n times k}.}

X A = B A ∈ K k × k , B ∈ K n × k .

This system of linear equations has one solution if and only if <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  is [invertible](https://en.wikipedia.org/wiki/Invertible_matrix#The_invertible_matrix_theorem)  .
This function assumes that <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  is invertible. 

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if the inputs are batches of matrices then
the output has the same batch dimensions. 

Letting *** be zero or more batch dimensions, 

* If `A`  has shape *(*, n, n)* and `B`  has shape *(*, n)* (a batch of vectors) or shape *(*, n, k)* (a batch of matrices or “multiple right-hand sides”), this function returns *X* of shape *(*, n)* or *(*, n, k)* respectively.
* Otherwise, if `A`  has shape *(*, n, n)* and `B`  has shape *(n,)* or *(n, k)* , `B`  is broadcasted to have shape *(*, n)* or *(*, n, k)* respectively.
This function then returns the solution of the resulting batch of systems of linear equations.

Note 

This function computes *X =*`A` *.inverse() @*`B`  in a faster and
more numerically stable way than performing the computations separately.

Note 

It is possible to compute the solution of the system <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             X
            </mi>
<mi>
             A
            </mi>
<mo>
             =
            </mo>
<mi>
             B
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            XA = B
           </annotation>
</semantics>
</math> -->X A = B XA = BX A = B  by passing the inputs `A`  and `B`  transposed and transposing the output returned by this function.

Note 

`A`  is allowed to be a non-batched *torch.sparse_csr_tensor* , but only with *left=True* .

Note 

When inputs are on a CUDA device, this function synchronizes that device with the CPU. For a version of this function that does not synchronize, see [`torch.linalg.solve_ex()`](torch.linalg.solve_ex.html#torch.linalg.solve_ex "torch.linalg.solve_ex")  .

See also 

[`torch.linalg.solve_triangular()`](torch.linalg.solve_triangular.html#torch.linalg.solve_triangular "torch.linalg.solve_triangular")  computes the solution of a triangular system of linear
equations with a unique solution.

Parameters
:   * **A** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, n, n)* where *** is zero or more batch dimensions.
* **B** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – right-hand side tensor of shape *(*, n)* or *(*, n, k)* or *(n,)* or *(n, k)* according to the rules described above

Keyword Arguments
:   * **left** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether to solve the system <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                A
               </mi>
<mi>
                X
               </mi>
<mo>
                =
               </mo>
<mi>
                B
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               AX=B
              </annotation>
</semantics>
</math> -->A X = B AX=BA X = B  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                X
               </mi>
<mi>
                A
               </mi>
<mo>
                =
               </mo>
<mi>
                B
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               XA = B
              </annotation>
</semantics>
</math> -->X A = B XA = BX A = B  . Default: *True* .

* **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – output tensor. Ignored if *None* . Default: *None* .

Raises
:   [**RuntimeError**](https://docs.python.org/3/library/exceptions.html#RuntimeError "(in Python v3.13)")  – if the `A`  matrix is not invertible or any matrix in a batched `A`  is not invertible.

Examples: 

```
>>> A = torch.randn(3, 3)
>>> b = torch.randn(3)
>>> x = torch.linalg.solve(A, b)
>>> torch.allclose(A @ x, b)
True
>>> A = torch.randn(2, 3, 3)
>>> B = torch.randn(2, 3, 4)
>>> X = torch.linalg.solve(A, B)
>>> X.shape
torch.Size([2, 3, 4])
>>> torch.allclose(A @ X, B)
True

>>> A = torch.randn(2, 3, 3)
>>> b = torch.randn(3, 1)
>>> x = torch.linalg.solve(A, b) # b is broadcasted to size (2, 3, 1)
>>> x.shape
torch.Size([2, 3, 1])
>>> torch.allclose(A @ x, b)
True
>>> b = torch.randn(3)
>>> x = torch.linalg.solve(A, b) # b is broadcasted to size (2, 3)
>>> x.shape
torch.Size([2, 3])
>>> Ax = A @ x.unsqueeze(-1)
>>> torch.allclose(Ax, b.unsqueeze(-1).expand_as(Ax))
True

```

