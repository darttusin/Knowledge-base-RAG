torch.Tensor.bernoulli 
================================================================================

Tensor. bernoulli ( *** , *generator = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a result tensor where each <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext mathvariant="monospace">
            result[i]
           </mtext>
</mrow>
<annotation encoding="application/x-tex">
           texttt{result[i]}
          </annotation>
</semantics>
</math> -->result[i] texttt{result[i]}result[i]  is independently
sampled from <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            Bernoulli
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mtext mathvariant="monospace">
            self[i]
           </mtext>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{Bernoulli}(texttt{self[i]})
          </annotation>
</semantics>
</math> -->Bernoulli ( self[i] ) text{Bernoulli}(texttt{self[i]})Bernoulli ( self[i] )  . `self`  must have
floating point `dtype`  , and the result will have the same `dtype`  . 

See [`torch.bernoulli()`](torch.bernoulli.html#torch.bernoulli "torch.bernoulli")

