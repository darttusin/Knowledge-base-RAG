torch.vander 
============================================================

torch. vander ( *x*  , *N = None*  , *increasing = False* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Generates a Vandermonde matrix. 

The columns of the output matrix are elementwise powers of the input vector <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
             x
            </mi>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mi>
              N
             </mi>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
<mo stretchy="false">
              )
             </mo>
</mrow>
</msup>
<mo separator="true">
            ,
           </mo>
<msup>
<mi>
             x
            </mi>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mi>
              N
             </mi>
<mo>
              −
             </mo>
<mn>
              2
             </mn>
<mo stretchy="false">
              )
             </mo>
</mrow>
</msup>
<mo separator="true">
            ,
           </mo>
<mi mathvariant="normal">
            .
           </mi>
<mi mathvariant="normal">
            .
           </mi>
<mi mathvariant="normal">
            .
           </mi>
<mo separator="true">
            ,
           </mo>
<msup>
<mi>
             x
            </mi>
<mn>
             0
            </mn>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           x^{(N-1)}, x^{(N-2)}, ..., x^0
          </annotation>
</semantics>
</math> -->x ( N − 1 ) , x ( N − 2 ) , . . . , x 0 x^{(N-1)}, x^{(N-2)}, ..., x^0x ( N − 1 ) , x ( N − 2 ) , ... , x 0  .
If increasing is True, the order of the columns is reversed <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
             x
            </mi>
<mn>
             0
            </mn>
</msup>
<mo separator="true">
            ,
           </mo>
<msup>
<mi>
             x
            </mi>
<mn>
             1
            </mn>
</msup>
<mo separator="true">
            ,
           </mo>
<mi mathvariant="normal">
            .
           </mi>
<mi mathvariant="normal">
            .
           </mi>
<mi mathvariant="normal">
            .
           </mi>
<mo separator="true">
            ,
           </mo>
<msup>
<mi>
             x
            </mi>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mi>
              N
             </mi>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
<mo stretchy="false">
              )
             </mo>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           x^0, x^1, ..., x^{(N-1)}
          </annotation>
</semantics>
</math> -->x 0 , x 1 , . . . , x ( N − 1 ) x^0, x^1, ..., x^{(N-1)}x 0 , x 1 , ... , x ( N − 1 )  . Such a
matrix with a geometric progression in each row is named for Alexandre-Theophile Vandermonde. 

Parameters
:   * **x** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – 1-D input tensor.
* **N** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – Number of columns in the output. If N is not specified,
a square array is returned <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo>
                =
               </mo>
<mi>
                l
               </mi>
<mi>
                e
               </mi>
<mi>
                n
               </mi>
<mo stretchy="false">
                (
               </mo>
<mi>
                x
               </mi>
<mo stretchy="false">
                )
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N = len(x))
              </annotation>
</semantics>
</math> -->( N = l e n ( x ) ) (N = len(x))( N = l e n ( x ))  .

* **increasing** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Order of the powers of the columns. If True,
the powers increase from left to right, if False (the default) they are reversed.

Returns
:   Vandermonde matrix. If increasing is False, the first column is <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
               x
              </mi>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo>
                −
               </mo>
<mn>
                1
               </mn>
<mo stretchy="false">
                )
               </mo>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
             x^{(N-1)}
            </annotation>
</semantics>
</math> -->x ( N − 1 ) x^{(N-1)}x ( N − 1 )  ,
the second <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
               x
              </mi>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo>
                −
               </mo>
<mn>
                2
               </mn>
<mo stretchy="false">
                )
               </mo>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
             x^{(N-2)}
            </annotation>
</semantics>
</math> -->x ( N − 2 ) x^{(N-2)}x ( N − 2 )  and so forth. If increasing is True, the columns
are <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
               x
              </mi>
<mn>
               0
              </mn>
</msup>
<mo separator="true">
              ,
             </mo>
<msup>
<mi>
               x
              </mi>
<mn>
               1
              </mn>
</msup>
<mo separator="true">
              ,
             </mo>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mo separator="true">
              ,
             </mo>
<msup>
<mi>
               x
              </mi>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo>
                −
               </mo>
<mn>
                1
               </mn>
<mo stretchy="false">
                )
               </mo>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
             x^0, x^1, ..., x^{(N-1)}
            </annotation>
</semantics>
</math> -->x 0 , x 1 , . . . , x ( N − 1 ) x^0, x^1, ..., x^{(N-1)}x 0 , x 1 , ... , x ( N − 1 )  .

Return type
:   [Tensor](../tensors.html#torch.Tensor "torch.Tensor")

Example: 

```
>>> x = torch.tensor([1, 2, 3, 5])
>>> torch.vander(x)
tensor([[  1,   1,   1,   1],
        [  8,   4,   2,   1],
        [ 27,   9,   3,   1],
        [125,  25,   5,   1]])
>>> torch.vander(x, N=3)
tensor([[ 1,  1,  1],
        [ 4,  2,  1],
        [ 9,  3,  1],
        [25,  5,  1]])
>>> torch.vander(x, N=3, increasing=True)
tensor([[ 1,  1,  1],
        [ 1,  2,  4],
        [ 1,  3,  9],
        [ 1,  5, 25]])

```

