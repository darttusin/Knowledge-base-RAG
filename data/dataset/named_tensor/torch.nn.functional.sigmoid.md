torch.nn.functional.sigmoid 
==========================================================================================

torch.nn.functional. sigmoid ( *input* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L2274) 
:   Applies the element-wise function <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            Sigmoid
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
<mn>
             1
            </mn>
<mrow>
<mn>
              1
             </mn>
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
           text{Sigmoid}(x) = frac{1}{1 + exp(-x)}
          </annotation>
</semantics>
</math> -->Sigmoid ( x ) = 1 1 + exp ⁡ ( − x ) text{Sigmoid}(x) = frac{1}{1 + exp(-x)}Sigmoid ( x ) = 1 + e x p ( − x ) 1 ​ 

See [`Sigmoid`](torch.nn.Sigmoid.html#torch.nn.Sigmoid "torch.nn.Sigmoid")  for more details.

