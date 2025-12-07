torch.nn.functional.softplus 
============================================================================================

torch.nn.functional. softplus ( *input*  , *beta = 1*  , *threshold = 20* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Applies element-wise, the function <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            Softplus
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
<mfrac>
<mn>
             1
            </mn>
<mi>
             β
            </mi>
</mfrac>
<mo>
            ∗
           </mo>
<mi>
            log
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
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
<mi>
            β
           </mi>
<mo>
            ∗
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{Softplus}(x) = frac{1}{beta} * log(1 + exp(beta * x))
          </annotation>
</semantics>
</math> -->Softplus ( x ) = 1 β ∗ log ⁡ ( 1 + exp ⁡ ( β ∗ x ) ) text{Softplus}(x) = frac{1}{beta} * log(1 + exp(beta * x))Softplus ( x ) = β 1 ​ ∗ lo g ( 1 + exp ( β ∗ x ))  . 

For numerical stability the implementation reverts to the linear function
when <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            i
           </mi>
<mi>
            n
           </mi>
<mi>
            p
           </mi>
<mi>
            u
           </mi>
<mi>
            t
           </mi>
<mo>
            ×
           </mo>
<mi>
            β
           </mi>
<mo>
            &gt;
           </mo>
<mi>
            t
           </mi>
<mi>
            h
           </mi>
<mi>
            r
           </mi>
<mi>
            e
           </mi>
<mi>
            s
           </mi>
<mi>
            h
           </mi>
<mi>
            o
           </mi>
<mi>
            l
           </mi>
<mi>
            d
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           input times beta &gt; threshold
          </annotation>
</semantics>
</math> -->i n p u t × β > t h r e s h o l d input times beta > thresholdin p u t × β > t h res h o l d  . 

See [`Softplus`](torch.nn.Softplus.html#torch.nn.Softplus "torch.nn.Softplus")  for more details.

