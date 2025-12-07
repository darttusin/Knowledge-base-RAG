torch.nn.functional.leaky_relu 
=================================================================================================

torch.nn.functional. leaky_relu ( *input*  , *negative_slope = 0.01*  , *inplace = False* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L1879) 
:   Applies element-wise, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            LeakyReLU
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
            max
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<mn>
            0
           </mn>
<mo separator="true">
            ,
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
<mtext>
            negative_slope
           </mtext>
<mo>
            ∗
           </mo>
<mi>
            min
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<mn>
            0
           </mn>
<mo separator="true">
            ,
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{LeakyReLU}(x) = max(0, x) + text{negative_slope} * min(0, x)
          </annotation>
</semantics>
</math> -->LeakyReLU ( x ) = max ⁡ ( 0 , x ) + negative_slope ∗ min ⁡ ( 0 , x ) text{LeakyReLU}(x) = max(0, x) + text{negative_slope} * min(0, x)LeakyReLU ( x ) = max ( 0 , x ) + negative_slope ∗ min ( 0 , x ) 

See [`LeakyReLU`](torch.nn.LeakyReLU.html#torch.nn.LeakyReLU "torch.nn.LeakyReLU")  for more details. 

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

