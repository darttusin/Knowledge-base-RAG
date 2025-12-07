torch.asinh 
==========================================================

torch. asinh ( *input : [Tensor](../tensors.html#torch.Tensor "torch.Tensor")*  , *** , *out : Optional [ [Tensor](../tensors.html#torch.Tensor "torch.Tensor") ]* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a new tensor with the inverse hyperbolic sine of the elements of `input`  . 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mtext>
             out
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo>
            =
           </mo>
<msup>
<mrow>
<mi>
              sinh
             </mi>
<mo>
              ⁡
             </mo>
</mrow>
<mrow>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
</mrow>
</msup>
<mo stretchy="false">
            (
           </mo>
<msub>
<mtext>
             input
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{out}_{i} = sinh^{-1}(text{input}_{i})
          </annotation>
</semantics>
</math> -->
out i = sinh ⁡ − 1 ( input i ) text{out}_{i} = sinh^{-1}(text{input}_{i})

out i ​ = sinh − 1 ( input i ​ )

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.randn(4)
>>> a
tensor([ 0.1606, -1.4267, -1.0899, -1.0250 ])
>>> torch.asinh(a)
tensor([ 0.1599, -1.1534, -0.9435, -0.8990 ])

```

