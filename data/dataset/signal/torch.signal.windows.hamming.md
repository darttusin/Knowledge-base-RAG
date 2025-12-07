torch.signal.windows.hamming 
============================================================================================

torch.signal.windows. hamming ( *M*  , *** , *sym = True*  , *dtype = None*  , *layout = torch.strided*  , *device = None*  , *requires_grad = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/signal/windows/windows.py#L421) 
:   Computes the Hamming window. 

The Hamming window is defined as follows: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             w
            </mi>
<mi>
             n
            </mi>
</msub>
<mo>
            =
           </mo>
<mi>
            α
           </mi>
<mo>
            −
           </mo>
<mi>
            β
           </mi>
<mtext>
</mtext>
<mi>
            cos
           </mi>
<mo>
            ⁡
           </mo>
<mrow>
<mo fence="true">
             (
            </mo>
<mfrac>
<mrow>
<mn>
               2
              </mn>
<mi>
               π
              </mi>
<mi>
               n
              </mi>
</mrow>
<mrow>
<mi>
               M
              </mi>
<mo>
               −
              </mo>
<mn>
               1
              </mn>
</mrow>
</mfrac>
<mo fence="true">
             )
            </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           w_n = alpha - beta cos left( frac{2 pi n}{M - 1} right)
          </annotation>
</semantics>
</math> -->
w n = α − β cos ⁡ ( 2 π n M − 1 ) w_n = alpha - beta cos left( frac{2 pi n}{M - 1} right)

w n ​ = α − β cos ( M − 1 2 πn ​ )

The window is normalized to 1 (maximum value is 1). However, the 1 doesn’t appear if `M`  is even and `sym`  is *True* . 

Parameters
: **M** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the length of the window.
In other words, the number of points of the returned window.

Keyword Arguments
:   * **sym** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If *False* , returns a periodic window suitable for use in spectral analysis.
If *True* , returns a symmetric window suitable for use in filter design. Default: *True* .
* **alpha** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – The coefficient <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                α
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               alpha
              </annotation>
</semantics>
</math> -->α alphaα  in the equation above.

* **beta** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – The coefficient <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                β
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               beta
              </annotation>
</semantics>
</math> -->β betaβ  in the equation above.

* **dtype** ( [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned tensor.
Default: if `None`  , uses a global default (see [`torch.set_default_dtype()`](torch.set_default_dtype.html#torch.set_default_dtype "torch.set_default_dtype")  ).
* **layout** ( [`torch.layout`](../tensor_attributes.html#torch.layout "torch.layout")  , optional) – the desired layout of returned Tensor.
Default: `torch.strided`  .
* **device** ( [`torch.device`](../tensor_attributes.html#torch.device "torch.device")  , optional) – the desired device of returned tensor.
Default: if `None`  , uses the current device for the default tensor type
(see [`torch.set_default_device()`](torch.set_default_device.html#torch.set_default_device "torch.set_default_device")  ). `device`  will be the CPU
for CPU tensor types and the current CUDA device for CUDA tensor types.
* **requires_grad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If autograd should record operations on the
returned tensor. Default: `False`  .

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

Examples: 

```
>>> # Generates a symmetric Hamming window.
>>> torch.signal.windows.hamming(10)
tensor([0.0800, 0.1876, 0.4601, 0.7700, 0.9723, 0.9723, 0.7700, 0.4601, 0.1876, 0.0800])

>>> # Generates a periodic Hamming window.
>>> torch.signal.windows.hamming(10, sym=False)
tensor([0.0800, 0.1679, 0.3979, 0.6821, 0.9121, 1.0000, 0.9121, 0.6821, 0.3979, 0.1679])

```

