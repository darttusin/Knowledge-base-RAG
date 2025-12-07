torch.cummax 
============================================================

torch. cummax ( *input*  , *dim*  , *** , *out = None* ) 
:   Returns a namedtuple `(values, indices)`  where `values`  is the cumulative maximum of
elements of `input`  in the dimension `dim`  . And `indices`  is the index
location of each maximum value found in the dimension `dim`  . 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             y
            </mi>
<mi>
             i
            </mi>
</msub>
<mo>
            =
           </mo>
<mi>
            m
           </mi>
<mi>
            a
           </mi>
<mi>
            x
           </mi>
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             x
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
             x
            </mi>
<mn>
             2
            </mn>
</msub>
<mo separator="true">
            ,
           </mo>
<msub>
<mi>
             x
            </mi>
<mn>
             3
            </mn>
</msub>
<mo separator="true">
            ,
           </mo>
<mo>
            …
           </mo>
<mo separator="true">
            ,
           </mo>
<msub>
<mi>
             x
            </mi>
<mi>
             i
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           y_i = max(x_1, x_2, x_3, dots, x_i)
          </annotation>
</semantics>
</math> -->
y i = m a x ( x 1 , x 2 , x 3 , … , x i ) y_i = max(x_1, x_2, x_3, dots, x_i)

y i ​ = ma x ( x 1 ​ , x 2 ​ , x 3 ​ , … , x i ​ )

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the dimension to do the operation over

Keyword Arguments
: **out** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – the result tuple of two output tensors (values, indices)

Example: 

```
>>> a = torch.randn(10)
>>> a
tensor([-0.3449, -1.5447,  0.0685, -1.5104, -1.1706,  0.2259,  1.4696, -1.3284,
     1.9946, -0.8209])
>>> torch.cummax(a, dim=0)
torch.return_types.cummax(
    values=tensor([-0.3449, -0.3449,  0.0685,  0.0685,  0.0685,  0.2259,  1.4696,  1.4696,
     1.9946,  1.9946]),
    indices=tensor([0, 0, 2, 2, 2, 5, 6, 6, 8, 8]))

```

