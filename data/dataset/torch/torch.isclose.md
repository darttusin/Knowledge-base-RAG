torch.isclose 
==============================================================

torch. isclose ( *input*  , *other*  , *rtol = 1e-05*  , *atol = 1e-08*  , *equal_nan = False* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a new tensor with boolean elements representing if each element of `input`  is “close” to the corresponding element of `other`  .
Closeness is defined as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            ∣
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
<msub>
<mtext>
             other
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo stretchy="false">
            ∣
           </mo>
<mo>
            ≤
           </mo>
<mtext mathvariant="monospace">
            rtol
           </mtext>
<mo>
            ×
           </mo>
<mo stretchy="false">
            ∣
           </mo>
<msub>
<mtext>
             other
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo stretchy="false">
            ∣
           </mo>
<mo>
            +
           </mo>
<mtext mathvariant="monospace">
            atol
           </mtext>
</mrow>
<annotation encoding="application/x-tex">
           lvert text{input}_i - text{other}_i rvert leq texttt{rtol} times lvert text{other}_i rvert + texttt{atol}
          </annotation>
</semantics>
</math> -->
∣ input i − other i ∣ ≤ rtol × ∣ other i ∣ + atol lvert text{input}_i - text{other}_i rvert leq texttt{rtol} times lvert text{other}_i rvert + texttt{atol}

∣ input i ​ − other i ​ ∣ ≤ rtol × ∣ other i ​ ∣ + atol

where `input`  and `other`  are finite. Where `input`  and/or `other`  are nonfinite they are close if and only if
they are equal, with NaNs being considered equal to each other when `equal_nan`  is True. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – first tensor to compare
* **other** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – second tensor to compare
* **rtol** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – relative tolerance. Default: 1e-05
* **atol** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – absolute tolerance. Default: 1e-08
* **equal_nan** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – if `True`  , then two `NaN`  s will be considered equal. Default: `False`

Examples: 

```
>>> torch.isclose(torch.tensor((1., 2, 3)), torch.tensor((1 + 1e-10, 3, 4)))
tensor([ True, False, False])
>>> torch.isclose(torch.tensor((float('inf'), 4)), torch.tensor((float('inf'), 6)), rtol=.5)
tensor([True, True])

```

