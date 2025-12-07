torch.linalg.ldl_factor 
===================================================================================

torch.linalg. ldl_factor ( *A*  , *** , *hermitian = False*  , *out = None* ) 
:   Computes a compact representation of the LDL factorization of a Hermitian or symmetric (possibly indefinite) matrix. 

When `A`  is complex valued it can be Hermitian ( `hermitian` *= True* )
or symmetric ( `hermitian` *= False* ). 

The factorization is of the form the form <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            A
           </mi>
<mo>
            =
           </mo>
<mi>
            L
           </mi>
<mi>
            D
           </mi>
<msup>
<mi>
             L
            </mi>
<mi>
             T
            </mi>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           A = L D L^T
          </annotation>
</semantics>
</math> -->A = L D L T A = L D L^TA = L D L T  .
If `hermitian`  is *True* then transpose operation is the conjugate transpose. 

<!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            L
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           L
          </annotation>
</semantics>
</math> -->L LL  (or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            U
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           U
          </annotation>
</semantics>
</math> -->U UU  ) and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            D
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           D
          </annotation>
</semantics>
</math> -->D DD  are stored in compact form in `LD`  .
They follow the format specified by [LAPACK’s sytrf](https://www.netlib.org/lapack/explore-html-3.6.1/d3/db6/group__double_s_ycomputational_gad91bde1212277b3e909eb6af7f64858a.html)  function.
These tensors may be used in [`torch.linalg.ldl_solve()`](torch.linalg.ldl_solve.html#torch.linalg.ldl_solve "torch.linalg.ldl_solve")  to solve linear systems. 

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if `A`  is a batch of matrices then
the output has the same batch dimensions. 

Note 

When inputs are on a CUDA device, this function synchronizes that device with the CPU. For a version of this function that does not synchronize, see [`torch.linalg.ldl_factor_ex()`](torch.linalg.ldl_factor_ex.html#torch.linalg.ldl_factor_ex "torch.linalg.ldl_factor_ex")  .

Parameters
: **A** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, n, n)* where *** is zero or more batch dimensions
consisting of symmetric or Hermitian matrices.

Keyword Arguments
:   * **hermitian** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether to consider the input to be Hermitian or symmetric.
For real-valued matrices, this switch has no effect. Default: *False* .
* **out** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – tuple of two tensors to write the output to. Ignored if *None* . Default: *None* .

Returns
:   A named tuple *(LD, pivots)* .

Examples: 

```
>>> A = torch.randn(3, 3)
>>> A = A @ A.mT # make symmetric
>>> A
tensor([[7.2079, 4.2414, 1.9428],
        [4.2414, 3.4554, 0.3264],
        [1.9428, 0.3264, 1.3823]])
>>> LD, pivots = torch.linalg.ldl_factor(A)
>>> LD
tensor([[ 7.2079,  0.0000,  0.0000],
        [ 0.5884,  0.9595,  0.0000],
        [ 0.2695, -0.8513,  0.1633]])
>>> pivots
tensor([1, 2, 3], dtype=torch.int32)

```

