torch.linalg.householder_product 
=====================================================================================================

torch.linalg. householder_product ( *A*  , *tau*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the first *n* columns of a product of Householder matrices. 

Let <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->C mathbb{C}C  , and
let <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
              m
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
           A in mathbb{K}^{m times n}
          </annotation>
</semantics>
</math> -->A ∈ K m × n A in mathbb{K}^{m times n}A ∈ K m × n  be a matrix with columns <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             a
            </mi>
<mi>
             i
            </mi>
</msub>
<mo>
            ∈
           </mo>
<msup>
<mi mathvariant="double-struck">
             K
            </mi>
<mi>
             m
            </mi>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           a_i in mathbb{K}^m
          </annotation>
</semantics>
</math> -->a i ∈ K m a_i in mathbb{K}^ma i ​ ∈ K m  for <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            i
           </mi>
<mo>
            =
           </mo>
<mn>
            1
           </mn>
<mo separator="true">
            ,
           </mo>
<mo>
            …
           </mo>
<mo separator="true">
            ,
           </mo>
<mi>
            m
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           i=1,ldots,m
          </annotation>
</semantics>
</math> -->i = 1 , … , m i=1,ldots,mi = 1 , … , m  with <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            m
           </mi>
<mo>
            ≥
           </mo>
<mi>
            n
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           m geq n
          </annotation>
</semantics>
</math> -->m ≥ n m geq nm ≥ n  . Denote by <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             b
            </mi>
<mi>
             i
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           b_i
          </annotation>
</semantics>
</math> -->b i b_ib i ​  the vector resulting from
zeroing out the first <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            i
           </mi>
<mo>
            −
           </mo>
<mn>
            1
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           i-1
          </annotation>
</semantics>
</math> -->i − 1 i-1i − 1  components of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             a
            </mi>
<mi>
             i
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           a_i
          </annotation>
</semantics>
</math> -->a i a_ia i ​  and setting to *1* the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            i
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           i
          </annotation>
</semantics>
</math> -->i ii  -th.
For a vector <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            τ
           </mi>
<mo>
            ∈
           </mo>
<msup>
<mi mathvariant="double-struck">
             K
            </mi>
<mi>
             k
            </mi>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           tau in mathbb{K}^k
          </annotation>
</semantics>
</math> -->τ ∈ K k tau in mathbb{K}^kτ ∈ K k  with <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            k
           </mi>
<mo>
            ≤
           </mo>
<mi>
            n
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           k leq n
          </annotation>
</semantics>
</math> -->k ≤ n k leq nk ≤ n  , this function computes the
first <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->n nn  columns of the matrix 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             H
            </mi>
<mn>
             1
            </mn>
</msub>
<msub>
<mi>
             H
            </mi>
<mn>
             2
            </mn>
</msub>
<mi mathvariant="normal">
            .
           </mi>
<mi mathvariant="normal">
            .
           </mi>
<mi mathvariant="normal">
            .
           </mi>
<msub>
<mi>
             H
            </mi>
<mi>
             k
            </mi>
</msub>
<mspace width="2em">
</mspace>
<mtext>
            with
           </mtext>
<mspace width="2em">
</mspace>
<msub>
<mi>
             H
            </mi>
<mi>
             i
            </mi>
</msub>
<mo>
            =
           </mo>
<msub>
<mi mathvariant="normal">
             I
            </mi>
<mi>
             m
            </mi>
</msub>
<mo>
            −
           </mo>
<msub>
<mi>
             τ
            </mi>
<mi>
             i
            </mi>
</msub>
<msub>
<mi>
             b
            </mi>
<mi>
             i
            </mi>
</msub>
<msubsup>
<mi>
             b
            </mi>
<mi>
             i
            </mi>
<mtext>
             H
            </mtext>
</msubsup>
</mrow>
<annotation encoding="application/x-tex">
           H_1H_2 ... H_k qquadtext{with}qquad H_i = mathrm{I}_m - tau_i b_i b_i^{text{H}}
          </annotation>
</semantics>
</math> -->
H 1 H 2 . . . H k with H i = I m − τ i b i b i H H_1H_2 ... H_k qquadtext{with}qquad H_i = mathrm{I}_m - tau_i b_i b_i^{text{H}}

H 1 ​ H 2 ​ ... H k ​ with H i ​ = I m ​ − τ i ​ b i ​ b i H ​

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi mathvariant="normal">
             I
            </mi>
<mi>
             m
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           mathrm{I}_m
          </annotation>
</semantics>
</math> -->I m mathrm{I}_mI m ​  is the *m* -dimensional identity matrix and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
             b
            </mi>
<mtext>
             H
            </mtext>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           b^{text{H}}
          </annotation>
</semantics>
</math> -->b H b^{text{H}}b H  is the
conjugate transpose when <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->b bb  is complex, and the transpose when <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->b bb  is real-valued.
The output matrix is the same size as the input matrix `A`  . 

See [Representation of Orthogonal or Unitary Matrices](https://www.netlib.org/lapack/lug/node128.html)  for further details. 

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if the inputs are batches of matrices then
the output has the same batch dimensions. 

See also 

[`torch.geqrf()`](torch.geqrf.html#torch.geqrf "torch.geqrf")  can be used together with this function to form the *Q* from the [`qr()`](torch.linalg.qr.html#torch.linalg.qr "torch.linalg.qr")  decomposition. 

[`torch.ormqr()`](torch.ormqr.html#torch.ormqr "torch.ormqr")  is a related function that computes the matrix multiplication
of a product of Householder matrices with another matrix.
However, that function is not supported by autograd.

Warning 

Gradient computations are only well-defined if <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              τ
             </mi>
<mi>
              i
             </mi>
</msub>
<mo mathvariant="normal">
             ≠
            </mo>
<mfrac>
<mn>
              1
             </mn>
<mrow>
<mi mathvariant="normal">
               ∣
              </mi>
<mi mathvariant="normal">
               ∣
              </mi>
<msub>
<mi>
                a
               </mi>
<mi>
                i
               </mi>
</msub>
<mi mathvariant="normal">
               ∣
              </mi>
<msup>
<mi mathvariant="normal">
                ∣
               </mi>
<mn>
                2
               </mn>
</msup>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
            tau_i neq frac{1}{||a_i||^2}
           </annotation>
</semantics>
</math> -->τ i ≠ 1 ∣ ∣ a i ∣ ∣ 2 tau_i neq frac{1}{||a_i||^2}τ i ​  = ∣∣ a i ​ ∣ ∣ 2 1 ​  .
If this condition is not met, no error will be thrown, but the gradient produced may contain *NaN* .

Parameters
:   * **A** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, m, n)* where *** is zero or more batch dimensions.
* **tau** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, k)* where *** is zero or more batch dimensions.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – output tensor. Ignored if *None* . Default: *None* .

Raises
:   [**RuntimeError**](https://docs.python.org/3/library/exceptions.html#RuntimeError "(in Python v3.13)")  – if `A`  doesn’t satisfy the requirement *m >= n* ,
 or `tau`  doesn’t satisfy the requirement *n >= k* .

Examples: 

```
>>> A = torch.randn(2, 2)
>>> h, tau = torch.geqrf(A)
>>> Q = torch.linalg.householder_product(h, tau)
>>> torch.dist(Q, torch.linalg.qr(A).Q)
tensor(0.)

>>> h = torch.randn(3, 2, 2, dtype=torch.complex128)
>>> tau = torch.randn(3, 1, dtype=torch.complex128)
>>> Q = torch.linalg.householder_product(h, tau)
>>> Q
tensor([[[ 1.8034+0.4184j,  0.2588-1.0174j],
        [-0.6853+0.7953j,  2.0790+0.5620j]],

        [[ 1.4581+1.6989j, -1.5360+0.1193j],
        [ 1.3877-0.6691j,  1.3512+1.3024j]],

        [[ 1.4766+0.5783j,  0.0361+0.6587j],
        [ 0.6396+0.1612j,  1.3693+0.4481j]]], dtype=torch.complex128)

```

