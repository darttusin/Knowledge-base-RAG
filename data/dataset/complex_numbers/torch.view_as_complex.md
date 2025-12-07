torch.view_as_complex 
================================================================================

torch. view_as_complex ( *input* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a view of `input`  as a complex tensor. For an input complex
tensor of `size` <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            m
           </mi>
<mn>
            1
           </mn>
<mo separator="true">
            ,
           </mo>
<mi>
            m
           </mi>
<mn>
            2
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
<mi>
            i
           </mi>
<mo separator="true">
            ,
           </mo>
<mn>
            2
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           m1, m2, dots, mi, 2
          </annotation>
</semantics>
</math> -->m 1 , m 2 , … , m i , 2 m1, m2, dots, mi, 2m 1 , m 2 , … , mi , 2  , this function returns a
new complex tensor of `size` <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            m
           </mi>
<mn>
            1
           </mn>
<mo separator="true">
            ,
           </mo>
<mi>
            m
           </mi>
<mn>
            2
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
<mi>
            i
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           m1, m2, dots, mi
          </annotation>
</semantics>
</math> -->m 1 , m 2 , … , m i m1, m2, dots, mim 1 , m 2 , … , mi  where the last
dimension of the input tensor is expected to represent the real and imaginary
components of complex numbers. 

Warning 

[`view_as_complex()`](#torch.view_as_complex "torch.view_as_complex")  is only supported for tensors with [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype") `torch.float64`  and `torch.float32`  . The input is
expected to have the last dimension of `size`  2. In addition, the
tensor must have a *stride* of 1 for its last dimension. The strides of all
other dimensions must be even numbers.

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Example: 

```
>>> x=torch.randn(4, 2)
>>> x
tensor([[ 1.6116, -0.5772],
        [-1.4606, -0.9120],
        [ 0.0786, -1.7497],
        [-0.6561, -1.6623]])
>>> torch.view_as_complex(x)
tensor([(1.6116-0.5772j), (-1.4606-0.9120j), (0.0786-1.7497j), (-0.6561-1.6623j)])

```

