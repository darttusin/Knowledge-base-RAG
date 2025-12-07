torch.cumsum 
============================================================

torch. cumsum ( *input*  , *dim*  , *** , *dtype = None*  , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns the cumulative sum of elements of `input`  in the dimension `dim`  . 

For example, if `input`  is a vector of size N, the result will also be
a vector of size N, with elements. 

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
<msub>
<mi>
             x
            </mi>
<mn>
             1
            </mn>
</msub>
<mo>
            +
           </mo>
<msub>
<mi>
             x
            </mi>
<mn>
             2
            </mn>
</msub>
<mo>
            +
           </mo>
<msub>
<mi>
             x
            </mi>
<mn>
             3
            </mn>
</msub>
<mo>
            +
           </mo>
<mo>
            ⋯
           </mo>
<mo>
            +
           </mo>
<msub>
<mi>
             x
            </mi>
<mi>
             i
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           y_i = x_1 + x_2 + x_3 + dots + x_i
          </annotation>
</semantics>
</math> -->
y i = x 1 + x 2 + x 3 + ⋯ + x i y_i = x_1 + x_2 + x_3 + dots + x_i

y i ​ = x 1 ​ + x 2 ​ + x 3 ​ + ⋯ + x i ​

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the dimension to do the operation over

Keyword Arguments
:   * **dtype** ( [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned tensor.
If specified, the input tensor is casted to [`dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  before the operation
is performed. This is useful for preventing data type overflows. Default: None.
* **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.randint(1, 20, (10,))
>>> a
tensor([13,  7,  3, 10, 13,  3, 15, 10,  9, 10])
>>> torch.cumsum(a, dim=0)
tensor([13, 20, 23, 33, 46, 49, 64, 74, 83, 93])

```

