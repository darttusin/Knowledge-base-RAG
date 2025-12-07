torch.cumprod 
==============================================================

torch. cumprod ( *input*  , *dim*  , *** , *dtype = None*  , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns the cumulative product of elements of `input`  in the dimension `dim`  . 

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
            ×
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
            ×
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
            ×
           </mo>
<mo>
            ⋯
           </mo>
<mo>
            ×
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
           y_i = x_1 times x_2times x_3times dots times x_i
          </annotation>
</semantics>
</math> -->
y i = x 1 × x 2 × x 3 × ⋯ × x i y_i = x_1 times x_2times x_3times dots times x_i

y i ​ = x 1 ​ × x 2 ​ × x 3 ​ × ⋯ × x i ​

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
>>> a = torch.randn(10)
>>> a
tensor([ 0.6001,  0.2069, -0.1919,  0.9792,  0.6727,  1.0062,  0.4126,
        -0.2129, -0.4206,  0.1968])
>>> torch.cumprod(a, dim=0)
tensor([ 0.6001,  0.1241, -0.0238, -0.0233, -0.0157, -0.0158, -0.0065,
         0.0014, -0.0006, -0.0001])

>>> a[5] = 0.0
>>> torch.cumprod(a, dim=0)
tensor([ 0.6001,  0.1241, -0.0238, -0.0233, -0.0157, -0.0000, -0.0000,
         0.0000, -0.0000, -0.0000])

```

