torch.Tensor.uniform_ 
==============================================================================

Tensor. uniform_ ( *from=0*  , *to=1*  , *** , *generator=None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Fills `self`  tensor with numbers sampled from the continuous uniform
distribution: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            f
           </mi>
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
<mtext>
              to
             </mtext>
<mo>
              −
             </mo>
<mtext>
              from
             </mtext>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           f(x) = dfrac{1}{text{to} - text{from}}
          </annotation>
</semantics>
</math> -->
f ( x ) = 1 to − from f(x) = dfrac{1}{text{to} - text{from}}

f ( x ) = to − from 1 ​

