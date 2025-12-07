torch.linalg.qr 
==================================================================

torch.linalg. qr ( *A*  , *mode = 'reduced'*  , *** , *out = None* ) 
:   Computes the QR decomposition of a matrix. 

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
the **full QR decomposition** of a matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A ∈ K m × n A in mathbb{K}^{m times n}A ∈ K m × n  is defined as 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            A
           </mi>
<mo>
            =
           </mo>
<mi>
            Q
           </mi>
<mi>
            R
           </mi>
<mpadded width="0px">
<mrow>
<mspace width="2em">
</mspace>
<mi>
              Q
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
                m
               </mi>
</mrow>
</msup>
<mo separator="true">
              ,
             </mo>
<mi>
              R
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
</mpadded>
</mrow>
<annotation encoding="application/x-tex">
           A = QRmathrlap{qquad Q in mathbb{K}^{m times m}, R in mathbb{K}^{m times n}}
          </annotation>
</semantics>
</math> -->
A = Q R Q ∈ K m × m , R ∈ K m × n A = QRmathrlap{qquad Q in mathbb{K}^{m times m}, R in mathbb{K}^{m times n}}

A = QR Q ∈ K m × m , R ∈ K m × n

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            Q
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           Q
          </annotation>
</semantics>
</math> -->Q QQ  is orthogonal in the real case and unitary in the complex case,
and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            R
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           R
          </annotation>
</semantics>
</math> -->R RR  is upper triangular with real diagonal (even in the complex case). 

When *m > n* (tall matrix), as *R* is upper triangular, its last *m - n* rows are zero.
In this case, we can drop the last *m - n* columns of *Q* to form the **reduced QR decomposition** : 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            A
           </mi>
<mo>
            =
           </mo>
<mi>
            Q
           </mi>
<mi>
            R
           </mi>
<mpadded width="0px">
<mrow>
<mspace width="2em">
</mspace>
<mi>
              Q
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
<mo separator="true">
              ,
             </mo>
<mi>
              R
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
</mpadded>
</mrow>
<annotation encoding="application/x-tex">
           A = QRmathrlap{qquad Q in mathbb{K}^{m times n}, R in mathbb{K}^{n times n}}
          </annotation>
</semantics>
</math> -->
A = Q R Q ∈ K m × n , R ∈ K n × n A = QRmathrlap{qquad Q in mathbb{K}^{m times n}, R in mathbb{K}^{n times n}}

A = QR Q ∈ K m × n , R ∈ K n × n

The reduced QR decomposition agrees with the full QR decomposition when *n >= m* (wide matrix). 

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if `A`  is a batch of matrices then
the output has the same batch dimensions. 

The parameter `mode`  chooses between the full and reduced QR decomposition.
If `A`  has shape *(*, m, n)* , denoting *k = min(m, n)*

* `mode` *= ‘reduced’* (default): Returns *(Q, R)* of shapes *(*, m, k)* , *(*, k, n)* respectively.
It is always differentiable.
* `mode` *= ‘complete’* : Returns *(Q, R)* of shapes *(*, m, m)* , *(*, m, n)* respectively.
It is differentiable for *m <= n* .
* `mode` *= ‘r’* : Computes only the reduced *R* . Returns *(Q, R)* with *Q* empty and *R* of shape *(*, k, n)* .
It is never differentiable.

Differences with *numpy.linalg.qr* : 

* `mode` *= ‘raw’* is not implemented.
* Unlike *numpy.linalg.qr* , this function always returns a tuple of two tensors.
When `mode` *= ‘r’* , the *Q* tensor is an empty tensor.

Warning 

The elements in the diagonal of *R* are not necessarily positive.
As such, the returned QR decomposition is only unique up to the sign of the diagonal of *R* .
Therefore, different platforms, like NumPy, or inputs on different devices,
may produce different valid decompositions.

Warning 

The QR decomposition is only well-defined if the first *k = min(m, n)* columns
of every matrix in `A`  are linearly independent.
If this condition is not met, no error will be thrown, but the QR produced
may be incorrect and its autodiff may fail or produce incorrect results.

Parameters
:   * **A** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – tensor of shape *(*, m, n)* where *** is zero or more batch dimensions.
* **mode** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – one of *‘reduced’* , *‘complete’* , *‘r’* .
Controls the shape of the returned tensors. Default: *‘reduced’* .

Keyword Arguments
: **out** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – output tuple of two tensors. Ignored if *None* . Default: *None* .

Returns
:   A named tuple *(Q, R)* .

Examples: 

```
>>> A = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
>>> Q, R = torch.linalg.qr(A)
>>> Q
tensor([[-0.8571,  0.3943,  0.3314],
        [-0.4286, -0.9029, -0.0343],
        [ 0.2857, -0.1714,  0.9429]])
>>> R
tensor([[ -14.0000,  -21.0000,   14.0000],
        [   0.0000, -175.0000,   70.0000],
        [   0.0000,    0.0000,  -35.0000]])
>>> (Q @ R).round()
tensor([[  12.,  -51.,    4.],
        [   6.,  167.,  -68.],
        [  -4.,   24.,  -41.]])
>>> (Q.T @ Q).round()
tensor([[ 1.,  0.,  0.],
        [ 0.,  1., -0.],
        [ 0., -0.,  1.]])
>>> Q2, R2 = torch.linalg.qr(A, mode='r')
>>> Q2
tensor([])
>>> torch.equal(R, R2)
True
>>> A = torch.randn(3, 4, 5)
>>> Q, R = torch.linalg.qr(A, mode='complete')
>>> torch.dist(Q @ R, A)
tensor(1.6099e-06)
>>> torch.dist(Q.mT @ Q, torch.eye(4))
tensor(6.2158e-07)

```

