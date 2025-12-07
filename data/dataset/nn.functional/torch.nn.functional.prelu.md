torch.nn.functional.prelu 
======================================================================================

torch.nn.functional. prelu ( *input*  , *weight* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Applies element-wise the function <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            PReLU
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
            weight
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
           text{PReLU}(x) = max(0,x) + text{weight} * min(0,x)
          </annotation>
</semantics>
</math> -->PReLU ( x ) = max ⁡ ( 0 , x ) + weight ∗ min ⁡ ( 0 , x ) text{PReLU}(x) = max(0,x) + text{weight} * min(0,x)PReLU ( x ) = max ( 0 , x ) + weight ∗ min ( 0 , x )  where weight is a
learnable parameter. 

Note 

*weight* is expected to be a scalar or 1-D tensor. If *weight* is 1-D,
its size must match the number of input channels, determined by *input.size(1)* when *input.dim() >= 2* , otherwise 1.
In the 1-D case, note that when *input* has dim > 2, *weight* can be expanded
to the shape of *input* in a way that is not possible using normal [broadcasting semantics](../notes/broadcasting.html#broadcasting-semantics)  .

See [`PReLU`](torch.nn.PReLU.html#torch.nn.PReLU "torch.nn.PReLU")  for more details.

