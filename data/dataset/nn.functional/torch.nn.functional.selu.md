torch.nn.functional.selu 
====================================================================================

torch.nn.functional. selu ( *input*  , *inplace = False* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L1817) 
:   Applies element-wise, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            SELU
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
            s
           </mi>
<mi>
            c
           </mi>
<mi>
            a
           </mi>
<mi>
            l
           </mi>
<mi>
            e
           </mi>
<mo>
            ∗
           </mo>
<mo stretchy="false">
            (
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
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{SELU}(x) = scale * (max(0,x) + min(0, alpha * (exp(x) - 1)))
          </annotation>
</semantics>
</math> -->SELU ( x ) = s c a l e ∗ ( max ⁡ ( 0 , x ) + min ⁡ ( 0 , α ∗ ( exp ⁡ ( x ) − 1 ) ) ) text{SELU}(x) = scale * (max(0,x) + min(0, alpha * (exp(x) - 1)))SELU ( x ) = sc a l e ∗ ( max ( 0 , x ) + min ( 0 , α ∗ ( exp ( x ) − 1 )))  ,
with <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            α
           </mi>
<mo>
            =
           </mo>
<mn>
            1.6732632423543772848170429916717
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           alpha=1.6732632423543772848170429916717
          </annotation>
</semantics>
</math> -->α = 1.6732632423543772848170429916717 alpha=1.6732632423543772848170429916717α = 1.6732632423543772848170429916717  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            s
           </mi>
<mi>
            c
           </mi>
<mi>
            a
           </mi>
<mi>
            l
           </mi>
<mi>
            e
           </mi>
<mo>
            =
           </mo>
<mn>
            1.0507009873554804934193349852946
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           scale=1.0507009873554804934193349852946
          </annotation>
</semantics>
</math> -->s c a l e = 1.0507009873554804934193349852946 scale=1.0507009873554804934193349852946sc a l e = 1.0507009873554804934193349852946  . 

See [`SELU`](torch.nn.SELU.html#torch.nn.SELU "torch.nn.SELU")  for more details. 

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

