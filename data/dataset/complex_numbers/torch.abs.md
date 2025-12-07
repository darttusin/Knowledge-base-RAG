torch.abs 
======================================================

torch. abs ( *input : [Tensor](../tensors.html#torch.Tensor "torch.Tensor")*  , *** , *out : Optional [ [Tensor](../tensors.html#torch.Tensor "torch.Tensor") ]* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the absolute value of each element in `input`  . 

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
<mi mathvariant="normal">
            ∣
           </mi>
<msub>
<mtext>
             input
            </mtext>
<mi>
             i
            </mi>
</msub>
<mi mathvariant="normal">
            ∣
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           text{out}_{i} = |text{input}_{i}|
          </annotation>
</semantics>
</math> -->
out i = ∣ input i ∣ text{out}_{i} = |text{input}_{i}|

out i ​ = ∣ input i ​ ∣

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.abs(torch.tensor([-1, -2, 3]))
tensor([ 1,  2,  3])

```

