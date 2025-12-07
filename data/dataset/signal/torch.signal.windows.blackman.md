torch.signal.windows.blackman 
==============================================================================================

torch.signal.windows. blackman ( *M*  , *** , *sym = True*  , *dtype = None*  , *layout = torch.strided*  , *device = None*  , *requires_grad = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/signal/windows/windows.py#L535) 
:   Computes the Blackman window. 

The Blackman window is defined as follows: 

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
<mn>
            0.42
           </mn>
<mo>
            −
           </mo>
<mn>
            0.5
           </mn>
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
<mo>
            +
           </mo>
<mn>
            0.08
           </mn>
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
               4
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
           w_n = 0.42 - 0.5 cos left( frac{2 pi n}{M - 1} right) + 0.08 cos left( frac{4 pi n}{M - 1} right)
          </annotation>
</semantics>
</math> -->
w n = 0.42 − 0.5 cos ⁡ ( 2 π n M − 1 ) + 0.08 cos ⁡ ( 4 π n M − 1 ) w_n = 0.42 - 0.5 cos left( frac{2 pi n}{M - 1} right) + 0.08 cos left( frac{4 pi n}{M - 1} right)

w n ​ = 0.42 − 0.5 cos ( M − 1 2 πn ​ ) + 0.08 cos ( M − 1 4 πn ​ )

The window is normalized to 1 (maximum value is 1). However, the 1 doesn’t appear if `M`  is even and `sym`  is *True* . 

Parameters
: **M** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the length of the window.
In other words, the number of points of the returned window.

Keyword Arguments
:   * **sym** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If *False* , returns a periodic window suitable for use in spectral analysis.
If *True* , returns a symmetric window suitable for use in filter design. Default: *True* .
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
>>> # Generates a symmetric Blackman window.
>>> torch.signal.windows.blackman(5)
tensor([-1.4901e-08,  3.4000e-01,  1.0000e+00,  3.4000e-01, -1.4901e-08])

>>> # Generates a periodic Blackman window.
>>> torch.signal.windows.blackman(5, sym=False)
tensor([-1.4901e-08,  2.0077e-01,  8.4923e-01,  8.4923e-01,  2.0077e-01])

```

