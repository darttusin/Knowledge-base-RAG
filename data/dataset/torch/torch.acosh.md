torch.acosh 
==========================================================

torch. acosh ( *input : [Tensor](../tensors.html#torch.Tensor "torch.Tensor")*  , *** , *out : Optional [ [Tensor](../tensors.html#torch.Tensor "torch.Tensor") ]* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a new tensor with the inverse hyperbolic cosine of the elements of `input`  . 

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
              cosh
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
           text{out}_{i} = cosh^{-1}(text{input}_{i})
          </annotation>
</semantics>
</math> -->
out i = cosh ⁡ − 1 ( input i ) text{out}_{i} = cosh^{-1}(text{input}_{i})

out i ​ = cosh − 1 ( input i ​ )

Note 

The domain of the inverse hyperbolic cosine is *[1, inf)* and values outside this range
will be mapped to `NaN`  , except for *+ INF* for which the output is mapped to *+ INF* .

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.randn(4).uniform_(1, 2)
>>> a
tensor([ 1.3192, 1.9915, 1.9674, 1.7151 ])
>>> torch.acosh(a)
tensor([ 0.7791, 1.3120, 1.2979, 1.1341 ])

```

