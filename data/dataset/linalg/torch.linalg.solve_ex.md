torch.linalg.solve_ex 
===============================================================================

torch.linalg. solve_ex ( *A*  , *B*  , *** , *left = True*  , *check_errors = False*  , *out = None* ) 
:   A version of [`solve()`](torch.linalg.solve.html#torch.linalg.solve "torch.linalg.solve")  that does not perform error checks unless `check_errors` *= True* .
It also returns the `info`  tensor returned by [LAPACK’s getrf](https://www.netlib.org/lapack/explore-html-3.6.1/dd/d9a/group__double_g_ecomputational_ga0019443faea08275ca60a734d0593e60.html)  . 

Note 

When the inputs are on a CUDA device, this function synchronizes only when `check_errors` *= True* .

Warning 

This function is “experimental” and it may change in a future PyTorch release.

Parameters
: **A** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, n, n)* where *** is zero or more batch dimensions.

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

* **check_errors** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – controls whether to check the content of `infos`  and raise
an error if it is non-zero. Default: *False* .
* **out** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – tuple of two tensors to write the output to. Ignored if *None* . Default: *None* .

Returns
:   A named tuple *(result, info)* .

Examples: 

```
>>> A = torch.randn(3, 3)
>>> Ainv, info = torch.linalg.solve_ex(A)
>>> torch.dist(torch.linalg.inv(A), Ainv)
tensor(0.)
>>> info
tensor(0, dtype=torch.int32)

```

