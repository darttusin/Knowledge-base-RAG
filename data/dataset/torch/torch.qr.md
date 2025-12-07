torch.qr 
====================================================

torch. qr ( *input : [Tensor](../tensors.html#torch.Tensor "torch.Tensor")*  , *some : [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") = True*  , *** , *out : Union [ [Tensor](../tensors.html#torch.Tensor "torch.Tensor") , Tuple [ [Tensor](../tensors.html#torch.Tensor "torch.Tensor") , ... ] , List [ [Tensor](../tensors.html#torch.Tensor "torch.Tensor") ] , [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") ]* ) 
:   Computes the QR decomposition of a matrix or a batch of matrices `input`  ,
and returns a namedtuple (Q, R) of tensors such that <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            input
           </mtext>
<mo>
            =
           </mo>
<mi>
            Q
           </mi>
<mi>
            R
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           text{input} = Q R
          </annotation>
</semantics>
</math> -->input = Q R text{input} = Q Rinput = QR  with <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->Q QQ  being an orthogonal matrix or batch of orthogonal matrices and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->R RR  being an upper triangular matrix or batch of upper triangular matrices. 

If `some`  is `True`  , then this function returns the thin (reduced) QR factorization.
Otherwise, if `some`  is `False`  , this function returns the complete QR factorization. 

Warning 

[`torch.qr()`](#torch.qr "torch.qr")  is deprecated in favor of [`torch.linalg.qr()`](torch.linalg.qr.html#torch.linalg.qr "torch.linalg.qr")  and will be removed in a future PyTorch release. The boolean parameter `some`  has been
replaced with a string parameter [`mode`](torch.mode.html#torch.mode "torch.mode")  . 

`Q, R = torch.qr(A)`  should be replaced with 

```
Q, R = torch.linalg.qr(A)

```

`Q, R = torch.qr(A, some=False)`  should be replaced with 

```
Q, R = torch.linalg.qr(A, mode="complete")

```

Warning 

If you plan to backpropagate through QR, note that the current backward implementation
is only well-defined when the first <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             min
            </mi>
<mo>
             ⁡
            </mo>
<mo stretchy="false">
             (
            </mo>
<mi>
             i
            </mi>
<mi>
             n
            </mi>
<mi>
             p
            </mi>
<mi>
             u
            </mi>
<mi>
             t
            </mi>
<mi mathvariant="normal">
             .
            </mi>
<mi>
             s
            </mi>
<mi>
             i
            </mi>
<mi>
             z
            </mi>
<mi>
             e
            </mi>
<mo stretchy="false">
             (
            </mo>
<mo>
             −
            </mo>
<mn>
             1
            </mn>
<mo stretchy="false">
             )
            </mo>
<mo separator="true">
             ,
            </mo>
<mi>
             i
            </mi>
<mi>
             n
            </mi>
<mi>
             p
            </mi>
<mi>
             u
            </mi>
<mi>
             t
            </mi>
<mi mathvariant="normal">
             .
            </mi>
<mi>
             s
            </mi>
<mi>
             i
            </mi>
<mi>
             z
            </mi>
<mi>
             e
            </mi>
<mo stretchy="false">
             (
            </mo>
<mo>
             −
            </mo>
<mn>
             2
            </mn>
<mo stretchy="false">
             )
            </mo>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            min(input.size(-1), input.size(-2))
           </annotation>
</semantics>
</math> -->min ⁡ ( i n p u t . s i z e ( − 1 ) , i n p u t . s i z e ( − 2 ) ) min(input.size(-1), input.size(-2))min ( in p u t . s i ze ( − 1 ) , in p u t . s i ze ( − 2 ))  columns of `input`  are linearly independent.
This behavior will probably change once QR supports pivoting.

Note 

This function uses LAPACK for CPU inputs and MAGMA for CUDA inputs,
and may produce different (valid) decompositions on different device types
or different platforms.

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                n
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (*, m, n)
              </annotation>
</semantics>
</math> -->( ∗ , m , n ) (*, m, n)( ∗ , m , n )  where *** is zero or more
batch dimensions consisting of matrices of dimension <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
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
<annotation encoding="application/x-tex">
               m times n
              </annotation>
</semantics>
</math> -->m × n m times nm × n  .

* **some** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) –

    Set to `True`  for reduced QR decomposition and `False`  for
        complete QR decomposition. If *k = min(m, n)* then:

    > + `some=True`  : returns *(Q, R)* with dimensions (m, k), (k, n) (default)
        > + `'some=False'`  : returns *(Q, R)* with dimensions (m, m), (m, n)

Keyword Arguments
: **out** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – tuple of *Q* and *R* tensors.
The dimensions of *Q* and *R* are detailed in the description of `some`  above.

Example: 

```
>>> a = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
>>> q, r = torch.qr(a)
>>> q
tensor([[-0.8571,  0.3943,  0.3314],
        [-0.4286, -0.9029, -0.0343],
        [ 0.2857, -0.1714,  0.9429]])
>>> r
tensor([[ -14.0000,  -21.0000,   14.0000],
        [   0.0000, -175.0000,   70.0000],
        [   0.0000,    0.0000,  -35.0000]])
>>> torch.mm(q, r).round()
tensor([[  12.,  -51.,    4.],
        [   6.,  167.,  -68.],
        [  -4.,   24.,  -41.]])
>>> torch.mm(q.t(), q).round()
tensor([[ 1.,  0.,  0.],
        [ 0.,  1., -0.],
        [ 0., -0.,  1.]])
>>> a = torch.randn(3, 4, 5)
>>> q, r = torch.qr(a, some=False)
>>> torch.allclose(torch.matmul(q, r), a)
True
>>> torch.allclose(torch.matmul(q.mT, q), torch.eye(5))
True

```

