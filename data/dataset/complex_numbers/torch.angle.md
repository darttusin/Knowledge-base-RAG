torch.angle 
==========================================================

torch. angle ( *input : [Tensor](../tensors.html#torch.Tensor "torch.Tensor")*  , *** , *out : Optional [ [Tensor](../tensors.html#torch.Tensor "torch.Tensor") ]* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the element-wise angle (in radians) of the given `input`  tensor. 

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
<mi>
            a
           </mi>
<mi>
            n
           </mi>
<mi>
            g
           </mi>
<mi>
            l
           </mi>
<mi>
            e
           </mi>
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
           text{out}_{i} = angle(text{input}_{i})
          </annotation>
</semantics>
</math> -->
out i = a n g l e ( input i ) text{out}_{i} = angle(text{input}_{i})

out i ​ = an g l e ( input i ​ )

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Note 

Starting in PyTorch 1.8, angle returns pi for negative real numbers,
zero for non-negative real numbers, and propagates NaNs. Previously
the function would return zero for all real numbers and not propagate
floating-point NaNs.

Example: 

```
>>> torch.angle(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]))*180/3.14159
tensor([ 135.,  135,  -45])

```

