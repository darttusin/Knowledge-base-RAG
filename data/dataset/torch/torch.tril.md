torch.tril 
========================================================

torch. tril ( *input*  , *diagonal = 0*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices `input`  , the other elements of the result tensor `out`  are set to 0. 

The lower triangular part of the matrix is defined as the elements on and
below the diagonal. 

The argument [`diagonal`](torch.diagonal.html#torch.diagonal "torch.diagonal")  controls which diagonal to consider. If [`diagonal`](torch.diagonal.html#torch.diagonal "torch.diagonal")  = 0, all elements on and below the main diagonal are
retained. A positive value includes just as many diagonals above the main
diagonal, and similarly a negative value excludes just as many diagonals below
the main diagonal. The main diagonal are the set of indices <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            {
           </mo>
<mo stretchy="false">
            (
           </mo>
<mi>
            i
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            i
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo stretchy="false">
            }
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           lbrace (i, i) rbrace
          </annotation>
</semantics>
</math> -->{ ( i , i ) } lbrace (i, i) rbrace{( i , i )}  for <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            i
           </mi>
<mo>
            ∈
           </mo>
<mo stretchy="false">
            [
           </mo>
<mn>
            0
           </mn>
<mo separator="true">
            ,
           </mo>
<mi>
            min
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            {
           </mo>
<msub>
<mi>
             d
            </mi>
<mn>
             1
            </mn>
</msub>
<mo separator="true">
            ,
           </mo>
<msub>
<mi>
             d
            </mi>
<mn>
             2
            </mn>
</msub>
<mo stretchy="false">
            }
           </mo>
<mo>
            −
           </mo>
<mn>
            1
           </mn>
<mo stretchy="false">
            ]
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           i in [0, min{d_{1}, d_{2}} - 1]
          </annotation>
</semantics>
</math> -->i ∈ [ 0 , min ⁡ { d 1 , d 2 } − 1 ] i in [0, min{d_{1}, d_{2}} - 1]i ∈ [ 0 , min { d 1 ​ , d 2 ​ } − 1 ]  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             d
            </mi>
<mn>
             1
            </mn>
</msub>
<mo separator="true">
            ,
           </mo>
<msub>
<mi>
             d
            </mi>
<mn>
             2
            </mn>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           d_{1}, d_{2}
          </annotation>
</semantics>
</math> -->d 1 , d 2 d_{1}, d_{2}d 1 ​ , d 2 ​  are the dimensions of the matrix. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **diagonal** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – the diagonal to consider

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.randn(3, 3)
>>> a
tensor([[-1.0813, -0.8619,  0.7105],
        [ 0.0935,  0.1380,  2.2112],
        [-0.3409, -0.9828,  0.0289]])
>>> torch.tril(a)
tensor([[-1.0813,  0.0000,  0.0000],
        [ 0.0935,  0.1380,  0.0000],
        [-0.3409, -0.9828,  0.0289]])

>>> b = torch.randn(4, 6)
>>> b
tensor([[ 1.2219,  0.5653, -0.2521, -0.2345,  1.2544,  0.3461],
        [ 0.4785, -0.4477,  0.6049,  0.6368,  0.8775,  0.7145],
        [ 1.1502,  3.2716, -1.1243, -0.5413,  0.3615,  0.6864],
        [-0.0614, -0.7344, -1.3164, -0.7648, -1.4024,  0.0978]])
>>> torch.tril(b, diagonal=1)
tensor([[ 1.2219,  0.5653,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.4785, -0.4477,  0.6049,  0.0000,  0.0000,  0.0000],
        [ 1.1502,  3.2716, -1.1243, -0.5413,  0.0000,  0.0000],
        [-0.0614, -0.7344, -1.3164, -0.7648, -1.4024,  0.0000]])
>>> torch.tril(b, diagonal=-1)
tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.4785,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 1.1502,  3.2716,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.0614, -0.7344, -1.3164,  0.0000,  0.0000,  0.0000]])

```

