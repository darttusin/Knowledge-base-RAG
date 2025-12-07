torch.nn.functional.celu 
====================================================================================

torch.nn.functional. celu ( *input*  , *alpha = 1.*  , *inplace = False* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L1846) 
:   Applies element-wise, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            CELU
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
            α
           </mi>
<mo>
            ∗
           </mo>
<mo stretchy="false">
            (
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
<mi>
            x
           </mi>
<mi mathvariant="normal">
            /
           </mi>
<mi>
            α
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            −
           </mo>
<mn>
            1
           </mn>
<mo stretchy="false">
            )
           </mo>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{CELU}(x) = max(0,x) + min(0, alpha * (exp(x/alpha) - 1))
          </annotation>
</semantics>
</math> -->CELU ( x ) = max ⁡ ( 0 , x ) + min ⁡ ( 0 , α ∗ ( exp ⁡ ( x / α ) − 1 ) ) text{CELU}(x) = max(0,x) + min(0, alpha * (exp(x/alpha) - 1))CELU ( x ) = max ( 0 , x ) + min ( 0 , α ∗ ( exp ( x / α ) − 1 ))  . 

See [`CELU`](torch.nn.CELU.html#torch.nn.CELU "torch.nn.CELU")  for more details. 

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

