torch.sgn 
======================================================

torch. sgn ( *input*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   This function is an extension of torch.sign() to complex tensors.
It computes a new tensor whose elements have
the same angles as the corresponding elements of `input`  and
absolute values (i.e. magnitudes) of one for complex tensors and
is equivalent to torch.sign() for non-complex tensors. 

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
<mrow>
<mo fence="true">
             {
            </mo>
<mtable columnalign="left left" columnspacing="1em" rowspacing="0.36em">
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mn>
                 0
                </mn>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
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
<mo>
                  =
                 </mo>
<mo>
                  =
                 </mo>
<mn>
                  0
                 </mn>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mfrac>
<msub>
<mtext>
                   input
                  </mtext>
<mi>
                   i
                  </mi>
</msub>
<mrow>
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
</mrow>
</mfrac>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mtext>
                 otherwise
                </mtext>
</mstyle>
</mtd>
</mtr>
</mtable>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           text{out}_{i} = begin{cases}
                0 &amp; |text{{input}}_i| == 0 
                frac{{text{{input}}_i}}{|{text{{input}}_i}|} &amp; text{otherwise}
                end{cases}
          </annotation>
</semantics>
</math> -->
out i = { 0 ∣ input i ∣ = = 0 input i ∣ input i ∣ otherwise text{out}_{i} = begin{cases}
 0 & |text{{input}}_i| == 0 
 frac{{text{{input}}_i}}{|{text{{input}}_i}|} & text{otherwise}
 end{cases}

out i ​ = { 0 ∣ input i ​ ∣ input i ​ ​ ​ ∣ input i ​ ∣ == 0 otherwise ​

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> t = torch.tensor([3+4j, 7-24j, 0, 1+2j])
>>> t.sgn()
tensor([0.6000+0.8000j, 0.2800-0.9600j, 0.0000+0.0000j, 0.4472+0.8944j])

```

