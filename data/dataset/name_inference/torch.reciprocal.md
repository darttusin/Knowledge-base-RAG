torch.reciprocal 
====================================================================

torch. reciprocal ( *input*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a new tensor with the reciprocal of the elements of `input` 

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
<mfrac>
<mn>
             1
            </mn>
<msub>
<mtext>
              input
             </mtext>
<mi>
              i
             </mi>
</msub>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           text{out}_{i} = frac{1}{text{input}_{i}}
          </annotation>
</semantics>
</math> -->
out i = 1 input i text{out}_{i} = frac{1}{text{input}_{i}}

out i ​ = input i ​ 1 ​

Note 

Unlike NumPy’s reciprocal, torch.reciprocal supports integral inputs. Integral
inputs to reciprocal are automatically [promoted](../tensor_attributes.html#type-promotion-doc)  to
the default scalar type.

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.randn(4)
>>> a
tensor([-0.4595, -2.1219, -1.4314,  0.7298])
>>> torch.reciprocal(a)
tensor([-2.1763, -0.4713, -0.6986,  1.3702])

```

