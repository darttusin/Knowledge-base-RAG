torch.lcm 
======================================================

torch. lcm ( *input*  , *other*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the element-wise least common multiple (LCM) of `input`  and `other`  . 

Both `input`  and `other`  must have integer types. 

Note 

This defines <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             l
            </mi>
<mi>
             c
            </mi>
<mi>
             m
            </mi>
<mo stretchy="false">
             (
            </mo>
<mn>
             0
            </mn>
<mo separator="true">
             ,
            </mo>
<mn>
             0
            </mn>
<mo stretchy="false">
             )
            </mo>
<mo>
             =
            </mo>
<mn>
             0
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            lcm(0, 0) = 0
           </annotation>
</semantics>
</math> -->l c m ( 0 , 0 ) = 0 lcm(0, 0) = 0l c m ( 0 , 0 ) = 0  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             l
            </mi>
<mi>
             c
            </mi>
<mi>
             m
            </mi>
<mo stretchy="false">
             (
            </mo>
<mn>
             0
            </mn>
<mo separator="true">
             ,
            </mo>
<mi>
             a
            </mi>
<mo stretchy="false">
             )
            </mo>
<mo>
             =
            </mo>
<mn>
             0
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            lcm(0, a) = 0
           </annotation>
</semantics>
</math> -->l c m ( 0 , a ) = 0 lcm(0, a) = 0l c m ( 0 , a ) = 0  .

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **other** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the second input tensor

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.tensor([5, 10, 15])
>>> b = torch.tensor([3, 4, 5])
>>> torch.lcm(a, b)
tensor([15, 20, 15])
>>> c = torch.tensor([3])
>>> torch.lcm(a, c)
tensor([15, 30, 15])

```

