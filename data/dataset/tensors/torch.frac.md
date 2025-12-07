torch.frac 
========================================================

torch. frac ( *input*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the fractional portion of each element in `input`  . 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mtext>
             out
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo>
            =
           </mo>
<msub>
<mtext>
             input
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo>
            −
           </mo>
<mrow>
<mo fence="true">
             ⌊
            </mo>
<mi mathvariant="normal">
             ∣
            </mi>
<msub>
<mtext>
              input
             </mtext>
<mi>
              i
             </mi>
</msub>
<mi mathvariant="normal">
             ∣
            </mi>
<mo fence="true">
             ⌋
            </mo>
</mrow>
<mo>
            ∗
           </mo>
<mi mathvariant="normal">
            sgn
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<msub>
<mtext>
             input
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{out}_{i} = text{input}_{i} - leftlfloor |text{input}_{i}| rightrfloor * operatorname{sgn}(text{input}_{i})
          </annotation>
</semantics>
</math> -->
out i = input i − ⌊ ∣ input i ∣ ⌋ ∗ sgn ⁡ ( input i ) text{out}_{i} = text{input}_{i} - leftlfloor |text{input}_{i}| rightrfloor * operatorname{sgn}(text{input}_{i})

out i ​ = input i ​ − ⌊ ∣ input i ​ ∣ ⌋ ∗ sgn ( input i ​ )

Example: 

```
>>> torch.frac(torch.tensor([1, 2.5, -3.2]))
tensor([ 0.0000,  0.5000, -0.2000])

```

