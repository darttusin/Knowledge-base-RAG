torch.cummin 
============================================================

torch. cummin ( *input*  , *dim*  , *** , *out = None* ) 
:   Returns a namedtuple `(values, indices)`  where `values`  is the cumulative minimum of
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
            i
           </mi>
<mi>
            n
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
           y_i = min(x_1, x_2, x_3, dots, x_i)
          </annotation>
</semantics>
</math> -->
y i = m i n ( x 1 , x 2 , x 3 , … , x i ) y_i = min(x_1, x_2, x_3, dots, x_i)

y i ​ = min ( x 1 ​ , x 2 ​ , x 3 ​ , … , x i ​ )

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the dimension to do the operation over

Keyword Arguments
: **out** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – the result tuple of two output tensors (values, indices)

Example: 

```
>>> a = torch.randn(10)
>>> a
tensor([-0.2284, -0.6628,  0.0975,  0.2680, -1.3298, -0.4220, -0.3885,  1.1762,
     0.9165,  1.6684])
>>> torch.cummin(a, dim=0)
torch.return_types.cummin(
    values=tensor([-0.2284, -0.6628, -0.6628, -0.6628, -1.3298, -1.3298, -1.3298, -1.3298,
    -1.3298, -1.3298]),
    indices=tensor([0, 1, 1, 1, 4, 4, 4, 4, 4, 4]))

```

