torch.nn.functional.tanh 
====================================================================================

torch.nn.functional. tanh ( *input* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L2263) 
:   Applies element-wise, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            Tanh
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
<mi>
            tanh
           </mi>
<mo>
            ⁡
           </mo>
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
<mrow>
<mi>
              exp
             </mi>
<mo>
              ⁡
             </mo>
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
              −
             </mo>
<mi>
              exp
             </mi>
<mo>
              ⁡
             </mo>
<mo stretchy="false">
              (
             </mo>
<mo>
              −
             </mo>
<mi>
              x
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
<mrow>
<mi>
              exp
             </mi>
<mo>
              ⁡
             </mo>
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
              +
             </mo>
<mi>
              exp
             </mi>
<mo>
              ⁡
             </mo>
<mo stretchy="false">
              (
             </mo>
<mo>
              −
             </mo>
<mi>
              x
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           text{Tanh}(x) = tanh(x) = frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}
          </annotation>
</semantics>
</math> -->Tanh ( x ) = tanh ⁡ ( x ) = exp ⁡ ( x ) − exp ⁡ ( − x ) exp ⁡ ( x ) + exp ⁡ ( − x ) text{Tanh}(x) = tanh(x) = frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}Tanh ( x ) = tanh ( x ) = e x p ( x ) + e x p ( − x ) e x p ( x ) − e x p ( − x ) ​ 

See [`Tanh`](torch.nn.Tanh.html#torch.nn.Tanh "torch.nn.Tanh")  for more details.

