torch.lu_solve 
=================================================================

torch. lu_solve ( *b*  , *LU_data*  , *LU_pivots*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns the LU solve of the linear system <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            A
           </mi>
<mi>
            x
           </mi>
<mo>
            =
           </mo>
<mi>
            b
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           Ax = b
          </annotation>
</semantics>
</math> -->A x = b Ax = bA x = b  using the partially pivoted
LU factorization of A from [`lu_factor()`](torch.linalg.lu_factor.html#torch.linalg.lu_factor "torch.linalg.lu_factor")  . 

This function supports `float`  , `double`  , `cfloat`  and `cdouble`  dtypes for `input`  . 

Warning 

[`torch.lu_solve()`](#torch.lu_solve "torch.lu_solve")  is deprecated in favor of [`torch.linalg.lu_solve()`](torch.linalg.lu_solve.html#torch.linalg.lu_solve "torch.linalg.lu_solve")  . [`torch.lu_solve()`](#torch.lu_solve "torch.lu_solve")  will be removed in a future PyTorch release. `X = torch.lu_solve(B, LU, pivots)`  should be replaced with 

```
X = linalg.lu_solve(LU, pivots, B)

```

Parameters
:   * **b** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the RHS tensor of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( ∗ , m , k ) (*, m, k)( ∗ , m , k )  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->∗ *∗  is zero or more batch dimensions.

* **LU_data** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the pivoted LU factorization of A from [`lu_factor()`](torch.linalg.lu_factor.html#torch.linalg.lu_factor "torch.linalg.lu_factor")  of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( ∗ , m , m ) (*, m, m)( ∗ , m , m )  ,
where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->∗ *∗  is zero or more batch dimensions.

* **LU_pivots** ( *IntTensor*  ) – the pivots of the LU factorization from [`lu_factor()`](torch.linalg.lu_factor.html#torch.linalg.lu_factor "torch.linalg.lu_factor")  of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (*, m)
              </annotation>
</semantics>
</math> -->( ∗ , m ) (*, m)( ∗ , m )  ,
where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->∗ *∗  is zero or more batch dimensions.
The batch dimensions of `LU_pivots`  must be equal to the batch dimensions of `LU_data`  .

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> A = torch.randn(2, 3, 3)
>>> b = torch.randn(2, 3, 1)
>>> LU, pivots = torch.linalg.lu_factor(A)
>>> x = torch.lu_solve(b, LU, pivots)
>>> torch.dist(A @ x, b)
tensor(1.00000e-07 *
       2.8312)

```

