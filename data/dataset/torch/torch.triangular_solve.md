torch.triangular_solve 
=================================================================================

torch. triangular_solve ( *b*  , *A*  , *upper = True*  , *transpose = False*  , *unitriangular = False*  , *** , *out = None* ) 
:   Solves a system of equations with a square upper or lower triangular invertible matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  and multiple right-hand sides <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            b
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           b
          </annotation>
</semantics>
</math> -->b bb  . 

In symbols, it solves <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
            b
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           AX = b
          </annotation>
</semantics>
</math> -->A X = b AX = bA X = b  and assumes <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  is square upper-triangular
(or lower-triangular if `upper` *= False* ) and does not have zeros on the diagonal. 

*torch.triangular_solve(b, A)* can take in 2D inputs *b, A* or inputs that are
batches of 2D matrices. If the inputs are batches, then returns
batched outputs *X*

If the diagonal of `A`  contains zeros or elements that are very close to zero and `unitriangular` *= False* (default) or if the input matrix is badly conditioned,
the result may contain *NaN* s. 

Supports input of float, double, cfloat and cdouble data types. 

Warning 

[`torch.triangular_solve()`](#torch.triangular_solve "torch.triangular_solve")  is deprecated in favor of [`torch.linalg.solve_triangular()`](torch.linalg.solve_triangular.html#torch.linalg.solve_triangular "torch.linalg.solve_triangular")  and will be removed in a future PyTorch release. [`torch.linalg.solve_triangular()`](torch.linalg.solve_triangular.html#torch.linalg.solve_triangular "torch.linalg.solve_triangular")  has its arguments reversed and does not return a
copy of one of the inputs. 

`X = torch.triangular_solve(B, A).solution`  should be replaced with 

```
X = torch.linalg.solve_triangular(A, B)

```

Parameters
:   * **b** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – multiple right-hand sides of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo separator="true">
                ,
               </mo>
<mi>
                m
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                k
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (*, m, k)
              </annotation>
</semantics>
</math> -->( ∗ , m , k ) (*, m, k)( ∗ , m , k )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
                ∗
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               *
              </annotation>
</semantics>
</math> -->∗ *∗  is zero of more batch dimensions

* **A** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input triangular coefficient matrix of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo separator="true">
                ,
               </mo>
<mi>
                m
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                m
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (*, m, m)
              </annotation>
</semantics>
</math> -->( ∗ , m , m ) (*, m, m)( ∗ , m , m )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
                ∗
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               *
              </annotation>
</semantics>
</math> -->∗ *∗  is zero or more batch dimensions

* **upper** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  is upper or lower triangular. Default: `True`  .

* **transpose** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – solves *op(A)X = b* where *op(A) = A^T* if this flag is `True`  ,
and *op(A) = A* if it is `False`  . Default: `False`  .
* **unitriangular** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  is unit triangular.
If True, the diagonal elements of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  are assumed to be
1 and not referenced from <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  . Default: `False`  .

Keyword Arguments
: **out** ( *(* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *)* *,* *optional*  ) – tuple of two tensors to write
the output to. Ignored if *None* . Default: *None* .

Returns
:   A namedtuple *(solution, cloned_coefficient)* where *cloned_coefficient* is a clone of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  and *solution* is the solution <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
              X
             </mi>
</mrow>
<annotation encoding="application/x-tex">
             X
            </annotation>
</semantics>
</math> -->X XX  to <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
              b
             </mi>
</mrow>
<annotation encoding="application/x-tex">
             AX = b
            </annotation>
</semantics>
</math> -->A X = b AX = bA X = b  (or whatever variant of the system of equations, depending on the keyword arguments.)

Examples: 

```
>>> A = torch.randn(2, 2).triu()
>>> A
tensor([[ 1.1527, -1.0753],
        [ 0.0000,  0.7986]])
>>> b = torch.randn(2, 3)
>>> b
tensor([[-0.0210,  2.3513, -1.5492],
        [ 1.5429,  0.7403, -1.0243]])
>>> torch.triangular_solve(b, A)
torch.return_types.triangular_solve(
solution=tensor([[ 1.7841,  2.9046, -2.5405],
        [ 1.9320,  0.9270, -1.2826]]),
cloned_coefficient=tensor([[ 1.1527, -1.0753],
        [ 0.0000,  0.7986]]))

```

