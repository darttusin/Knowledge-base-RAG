torch.Tensor.bernoulli_ 
==================================================================================

Tensor. bernoulli_ ( *p = 0.5*  , *** , *generator = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Fills each location of `self`  with an independent sample from <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            Bernoulli
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mtext mathvariant="monospace">
            p
           </mtext>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{Bernoulli}(texttt{p})
          </annotation>
</semantics>
</math> -->Bernoulli ( p ) text{Bernoulli}(texttt{p})Bernoulli ( p )  . `self`  can have integral `dtype`  . 

`p`  should either be a scalar or tensor containing probabilities to be
used for drawing the binary random number. 

If it is a tensor, the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mtext>
             i
            </mtext>
<mrow>
<mi>
              t
             </mi>
<mi>
              h
             </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           text{i}^{th}
          </annotation>
</semantics>
</math> -->i t h text{i}^{th}i t h  element of `self`  tensor
will be set to a value sampled from <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            Bernoulli
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mtext mathvariant="monospace">
            p_tensor[i]
           </mtext>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{Bernoulli}(texttt{p_tensor[i]})
          </annotation>
</semantics>
</math> -->Bernoulli ( p_tensor[i] ) text{Bernoulli}(texttt{p_tensor[i]})Bernoulli ( p_tensor[i] )  . In this case *p* must have
floating point `dtype`  . 

See also [`bernoulli()`](torch.Tensor.bernoulli.html#torch.Tensor.bernoulli "torch.Tensor.bernoulli")  and [`torch.bernoulli()`](torch.bernoulli.html#torch.bernoulli "torch.bernoulli")

