torch.Tensor.exponential_ 
======================================================================================

Tensor. exponential_ ( *lambd = 1*  , *** , *generator = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Fills `self`  tensor with elements drawn from the PDF (probability density function): 

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
<mi>
            λ
           </mi>
<msup>
<mi>
             e
            </mi>
<mrow>
<mo>
              −
             </mo>
<mi>
              λ
             </mi>
<mi>
              x
             </mi>
</mrow>
</msup>
<mo separator="true">
            ,
           </mo>
<mi>
            x
           </mi>
<mo>
            &gt;
           </mo>
<mn>
            0
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           f(x) = lambda e^{-lambda x}, x &gt; 0
          </annotation>
</semantics>
</math> -->
f ( x ) = λ e − λ x , x > 0 f(x) = lambda e^{-lambda x}, x > 0

f ( x ) = λ e − λ x , x > 0

Note 

In probability theory, exponential distribution is supported on interval [0, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             inf
            </mi>
<mo>
             ⁡
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            inf
           </annotation>
</semantics>
</math> -->inf ⁡ infin f  ) (i.e., <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             x
            </mi>
<mo>
             &gt;
            </mo>
<mo>
             =
            </mo>
<mn>
             0
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            x &gt;= 0
           </annotation>
</semantics>
</math> -->x > = 0 x >= 0x >= 0  )
implying that zero can be sampled from the exponential distribution.
However, [`torch.Tensor.exponential_()`](#torch.Tensor.exponential_ "torch.Tensor.exponential_")  does not sample zero,
which means that its actual support is the interval (0, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             inf
            </mi>
<mo>
             ⁡
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            inf
           </annotation>
</semantics>
</math> -->inf ⁡ infin f  ). 

Note that [`torch.distributions.exponential.Exponential()`](../distributions.html#torch.distributions.exponential.Exponential "torch.distributions.exponential.Exponential")  is supported on the interval [0, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             inf
            </mi>
<mo>
             ⁡
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            inf
           </annotation>
</semantics>
</math> -->inf ⁡ infin f  ) and can sample zero.

