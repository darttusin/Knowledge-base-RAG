torch.logspace 
================================================================

torch. logspace ( *start*  , *end*  , *steps*  , *base = 10.0*  , *** , *out = None*  , *dtype = None*  , *layout = torch.strided*  , *device = None*  , *requires_grad = False* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Creates a one-dimensional tensor of size `steps`  whose values are evenly
spaced from <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mtext>
             base
            </mtext>
<mtext>
             start
            </mtext>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           {{text{{base}}}}^{{text{{start}}}}
          </annotation>
</semantics>
</math> -->base start {{text{{base}}}}^{{text{{start}}}}base start  to <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mtext>
             base
            </mtext>
<mtext>
             end
            </mtext>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           {{text{{base}}}}^{{text{{end}}}}
          </annotation>
</semantics>
</math> -->base end {{text{{base}}}}^{{text{{end}}}}base end  , inclusive, on a logarithmic scale
with base `base`  . That is, the values are: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<msup>
<mtext>
             base
            </mtext>
<mtext>
             start
            </mtext>
</msup>
<mo separator="true">
            ,
           </mo>
<msup>
<mtext>
             base
            </mtext>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mtext>
              start
             </mtext>
<mo>
              +
             </mo>
<mfrac>
<mrow>
<mtext>
                end
               </mtext>
<mo>
                −
               </mo>
<mtext>
                start
               </mtext>
</mrow>
<mrow>
<mtext>
                steps
               </mtext>
<mo>
                −
               </mo>
<mn>
                1
               </mn>
</mrow>
</mfrac>
<mo stretchy="false">
              )
             </mo>
</mrow>
</msup>
<mo separator="true">
            ,
           </mo>
<mo>
            …
           </mo>
<mo separator="true">
            ,
           </mo>
<msup>
<mtext>
             base
            </mtext>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mtext>
              start
             </mtext>
<mo>
              +
             </mo>
<mo stretchy="false">
              (
             </mo>
<mtext>
              steps
             </mtext>
<mo>
              −
             </mo>
<mn>
              2
             </mn>
<mo stretchy="false">
              )
             </mo>
<mo>
              ∗
             </mo>
<mfrac>
<mrow>
<mtext>
                end
               </mtext>
<mo>
                −
               </mo>
<mtext>
                start
               </mtext>
</mrow>
<mrow>
<mtext>
                steps
               </mtext>
<mo>
                −
               </mo>
<mn>
                1
               </mn>
</mrow>
</mfrac>
<mo stretchy="false">
              )
             </mo>
</mrow>
</msup>
<mo separator="true">
            ,
           </mo>
<msup>
<mtext>
             base
            </mtext>
<mtext>
             end
            </mtext>
</msup>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (text{base}^{text{start}},
text{base}^{(text{start} + frac{text{end} - text{start}}{ text{steps} - 1})},
ldots,
text{base}^{(text{start} + (text{steps} - 2) * frac{text{end} - text{start}}{ text{steps} - 1})},
text{base}^{text{end}})
          </annotation>
</semantics>
</math> -->
( base start , base ( start + end − start steps − 1 ) , … , base ( start + ( steps − 2 ) ∗ end − start steps − 1 ) , base end ) (text{base}^{text{start}},
text{base}^{(text{start} + frac{text{end} - text{start}}{ text{steps} - 1})},
ldots,
text{base}^{(text{start} + (text{steps} - 2) * frac{text{end} - text{start}}{ text{steps} - 1})},
text{base}^{text{end}})

( base start , base ( start + steps − 1 end − start ​ ) , … , base ( start + ( steps − 2 ) ∗ steps − 1 end − start ​ ) , base end )

From PyTorch 1.11 logspace requires the steps argument. Use steps=100 to restore the previous behavior. 

Parameters
:   * **start** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *or* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the starting value for the set of points. If *Tensor* , it must be 0-dimensional
* **end** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *or* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the ending value for the set of points. If *Tensor* , it must be 0-dimensional
* **steps** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – size of the constructed tensor
* **base** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – base of the logarithm function. Default: `10.0`  .

Keyword Arguments
:   * **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.
* **dtype** ( [*torch.dtype*](../tensor_attributes.html#torch.dtype "torch.dtype") *,* *optional*  ) – the data type to perform the computation in.
Default: if None, uses the global default dtype (see torch.get_default_dtype())
when both `start`  and `end`  are real,
and corresponding complex dtype when either is complex.
* **layout** ( [`torch.layout`](../tensor_attributes.html#torch.layout "torch.layout")  , optional) – the desired layout of returned Tensor.
Default: `torch.strided`  .
* **device** ( [`torch.device`](../tensor_attributes.html#torch.device "torch.device")  , optional) – the desired device of returned tensor.
Default: if `None`  , uses the current device for the default tensor type
(see [`torch.set_default_device()`](torch.set_default_device.html#torch.set_default_device "torch.set_default_device")  ). [`device`](../tensor_attributes.html#torch.device "torch.device")  will be the CPU
for CPU tensor types and the current CUDA device for CUDA tensor types.
* **requires_grad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If autograd should record operations on the
returned tensor. Default: `False`  .

Example: 

```
>>> torch.logspace(start=-10, end=10, steps=5)
tensor([ 1.0000e-10,  1.0000e-05,  1.0000e+00,  1.0000e+05,  1.0000e+10])
>>> torch.logspace(start=0.1, end=1.0, steps=5)
tensor([  1.2589,   2.1135,   3.5481,   5.9566,  10.0000])
>>> torch.logspace(start=0.1, end=1.0, steps=1)
tensor([1.2589])
>>> torch.logspace(start=2, end=2, steps=1, base=2)
tensor([4.0])

```

