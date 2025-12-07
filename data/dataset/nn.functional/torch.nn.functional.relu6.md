torch.nn.functional.relu6 
======================================================================================

torch.nn.functional. relu6 ( *input*  , *inplace = False* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L1777) 
:   Applies the element-wise function <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            ReLU6
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
            min
           </mi>
<mo>
            ⁡
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
<mo separator="true">
            ,
           </mo>
<mn>
            6
           </mn>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{ReLU6}(x) = min(max(0,x), 6)
          </annotation>
</semantics>
</math> -->ReLU6 ( x ) = min ⁡ ( max ⁡ ( 0 , x ) , 6 ) text{ReLU6}(x) = min(max(0,x), 6)ReLU6 ( x ) = min ( max ( 0 , x ) , 6 )  . 

See [`ReLU6`](torch.nn.ReLU6.html#torch.nn.ReLU6 "torch.nn.ReLU6")  for more details. 

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

