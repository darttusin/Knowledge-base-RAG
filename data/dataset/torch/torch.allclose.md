torch.allclose 
================================================================

torch. allclose ( *input : [Tensor](../tensors.html#torch.Tensor "torch.Tensor")*  , *other : [Tensor](../tensors.html#torch.Tensor "torch.Tensor")*  , *rtol : [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") = 1e-05*  , *atol : [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") = 1e-08*  , *equal_nan : [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") = False* ) → [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") 
:   This function checks if `input`  and `other`  satisfy the condition: 

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
            atol
           </mtext>
<mo>
            +
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
</mrow>
<annotation encoding="application/x-tex">
           lvert text{input}_i - text{other}_i rvert leq texttt{atol} + texttt{rtol} times lvert text{other}_i rvert
          </annotation>
</semantics>
</math> -->
∣ input i − other i ∣ ≤ atol + rtol × ∣ other i ∣ lvert text{input}_i - text{other}_i rvert leq texttt{atol} + texttt{rtol} times lvert text{other}_i rvert

∣ input i ​ − other i ​ ∣ ≤ atol + rtol × ∣ other i ​ ∣

elementwise, for all elements of `input`  and `other`  . The behaviour of this function is analogous to [numpy.allclose](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html) 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – first tensor to compare
* **other** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – second tensor to compare
* **atol** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – absolute tolerance. Default: 1e-08
* **rtol** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – relative tolerance. Default: 1e-05
* **equal_nan** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – if `True`  , then two `NaN`  s will be considered equal. Default: `False`

Example: 

```
>>> torch.allclose(torch.tensor([10000., 1e-07]), torch.tensor([10000.1, 1e-08]))
False
>>> torch.allclose(torch.tensor([10000., 1e-08]), torch.tensor([10000.1, 1e-09]))
True
>>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]))
False
>>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]), equal_nan=True)
True

```

