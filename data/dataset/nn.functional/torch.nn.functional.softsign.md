torch.nn.functional.softsign 
============================================================================================

torch.nn.functional. softsign ( *input* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L2027) 
:   Applies element-wise, the function <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            SoftSign
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mfrac>
<mi>
             x
            </mi>
<mrow>
<mn>
              1
             </mn>
<mo>
              +
             </mo>
<mi mathvariant="normal">
              ∣
             </mi>
<mi>
              x
             </mi>
<mi mathvariant="normal">
              ∣
             </mi>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           text{SoftSign}(x) = frac{x}{1 + |x|}
          </annotation>
</semantics>
</math> -->SoftSign ( x ) = x 1 + ∣ x ∣ text{SoftSign}(x) = frac{x}{1 + |x|}SoftSign ( x ) = 1 + ∣ x ∣ x ​ 

See [`Softsign`](torch.nn.Softsign.html#torch.nn.Softsign "torch.nn.Softsign")  for more details.

