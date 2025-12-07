torch.lgamma 
============================================================

torch. lgamma ( *input*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the natural logarithm of the absolute value of the gamma function on `input`  . 

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
            ln
           </mi>
<mo>
            ⁡
           </mo>
<mi mathvariant="normal">
            ∣
           </mi>
<mi mathvariant="normal">
            Γ
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
<mi mathvariant="normal">
            ∣
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           text{out}_{i} = ln |Gamma(text{input}_{i})|
          </annotation>
</semantics>
</math> -->
out i = ln ⁡ ∣ Γ ( input i ) ∣ text{out}_{i} = ln |Gamma(text{input}_{i})|

out i ​ = ln ∣Γ ( input i ​ ) ∣

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.arange(0.5, 2, 0.5)
>>> torch.lgamma(a)
tensor([ 0.5724,  0.0000, -0.1208])

```

