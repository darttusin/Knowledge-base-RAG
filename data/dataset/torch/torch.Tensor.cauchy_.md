torch.Tensor.cauchy_ 
============================================================================

Tensor. cauchy_ ( *median = 0*  , *sigma = 1*  , *** , *generator = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Fills the tensor with numbers drawn from the Cauchy distribution: 

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
<mi>
             π
            </mi>
</mfrac>
<mfrac>
<mi>
             σ
            </mi>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mi>
              x
             </mi>
<mo>
              −
             </mo>
<mtext>
              median
             </mtext>
<msup>
<mo stretchy="false">
               )
              </mo>
<mn>
               2
              </mn>
</msup>
<mo>
              +
             </mo>
<msup>
<mi>
               σ
              </mi>
<mn>
               2
              </mn>
</msup>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           f(x) = dfrac{1}{pi} dfrac{sigma}{(x - text{median})^2 + sigma^2}
          </annotation>
</semantics>
</math> -->
f ( x ) = 1 π σ ( x − median ) 2 + σ 2 f(x) = dfrac{1}{pi} dfrac{sigma}{(x - text{median})^2 + sigma^2}

f ( x ) = π 1 ​ ( x − median ) 2 + σ 2 σ ​

Note 

Sigma ( <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             σ
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            sigma
           </annotation>
</semantics>
</math> -->σ sigmaσ  ) is used to denote the scale parameter in Cauchy distribution.

