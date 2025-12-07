torch.nn.functional.logsigmoid 
================================================================================================

torch.nn.functional. logsigmoid ( *input* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Applies element-wise <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            LogSigmoid
           </mtext>
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             x
            </mi>
<mi>
             i
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mi>
            log
           </mi>
<mo>
            ⁡
           </mo>
<mrow>
<mo fence="true">
             (
            </mo>
<mfrac>
<mn>
              1
             </mn>
<mrow>
<mn>
               1
              </mn>
<mo>
               +
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
<mo>
               −
              </mo>
<msub>
<mi>
                x
               </mi>
<mi>
                i
               </mi>
</msub>
<mo stretchy="false">
               )
              </mo>
</mrow>
</mfrac>
<mo fence="true">
             )
            </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           text{LogSigmoid}(x_i) = log left(frac{1}{1 + exp(-x_i)}right)
          </annotation>
</semantics>
</math> -->LogSigmoid ( x i ) = log ⁡ ( 1 1 + exp ⁡ ( − x i ) ) text{LogSigmoid}(x_i) = log left(frac{1}{1 + exp(-x_i)}right)LogSigmoid ( x i ​ ) = lo g ( 1 + e x p ( − x i ​ ) 1 ​ ) 

See [`LogSigmoid`](torch.nn.LogSigmoid.html#torch.nn.LogSigmoid "torch.nn.LogSigmoid")  for more details.

