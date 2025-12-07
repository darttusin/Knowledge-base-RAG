torch.view_as_real 
==========================================================================

torch. view_as_real ( *input* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a view of `input`  as a real tensor. For an input complex tensor of `size` <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->m 1 , m 2 , … , m i m1, m2, dots, mim 1 , m 2 , … , mi  , this function returns a new
real tensor of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->m 1 , m 2 , … , m i , 2 m1, m2, dots, mi, 2m 1 , m 2 , … , mi , 2  , where the last dimension of size 2
represents the real and imaginary components of complex numbers. 

Warning 

[`view_as_real()`](#torch.view_as_real "torch.view_as_real")  is only supported for tensors with `complex dtypes`  .

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Example: 

```
>>> x=torch.randn(4, dtype=torch.cfloat)
>>> x
tensor([(0.4737-0.3839j), (-0.2098-0.6699j), (0.3470-0.9451j), (-0.5174-1.3136j)])
>>> torch.view_as_real(x)
tensor([[ 0.4737, -0.3839],
        [-0.2098, -0.6699],
        [ 0.3470, -0.9451],
        [-0.5174, -1.3136]])

```

