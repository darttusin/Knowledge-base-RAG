torch.logaddexp2 
====================================================================

torch. logaddexp2 ( *input*  , *other*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Logarithm of the sum of exponentiations of the inputs in base-2. 

Calculates pointwise <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mrow>
<mi>
              log
             </mi>
<mo>
              ⁡
             </mo>
</mrow>
<mn>
             2
            </mn>
</msub>
<mrow>
<mo fence="true">
             (
            </mo>
<msup>
<mn>
              2
             </mn>
<mi>
              x
             </mi>
</msup>
<mo>
             +
            </mo>
<msup>
<mn>
              2
             </mn>
<mi>
              y
             </mi>
</msup>
<mo fence="true">
             )
            </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           log_2left(2^x + 2^yright)
          </annotation>
</semantics>
</math> -->log ⁡ 2 ( 2 x + 2 y ) log_2left(2^x + 2^yright)lo g 2 ​ ( 2 x + 2 y )  . See [`torch.logaddexp()`](torch.logaddexp.html#torch.logaddexp "torch.logaddexp")  for more details. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **other** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the second input tensor

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

