torch.atanh 
==========================================================

torch. atanh ( *input : [Tensor](../tensors.html#torch.Tensor "torch.Tensor")*  , *** , *out : Optional [ [Tensor](../tensors.html#torch.Tensor "torch.Tensor") ]* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a new tensor with the inverse hyperbolic tangent of the elements of `input`  . 

Note 

The domain of the inverse hyperbolic tangent is *(-1, 1)* and values outside this range
will be mapped to `NaN`  , except for the values *1* and *-1* for which the output is
mapped to *+/-INF* respectively.

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
              tanh
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
           text{out}_{i} = tanh^{-1}(text{input}_{i})
          </annotation>
</semantics>
</math> -->
out i = tanh ⁡ − 1 ( input i ) text{out}_{i} = tanh^{-1}(text{input}_{i})

out i ​ = tanh − 1 ( input i ​ )

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.randn(4).uniform_(-1, 1)
>>> a
tensor([ -0.9385, 0.2968, -0.8591, -0.1871 ])
>>> torch.atanh(a)
tensor([ -1.7253, 0.3060, -1.2899, -0.1893 ])

```

