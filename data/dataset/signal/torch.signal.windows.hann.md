torch.signal.windows.hann 
======================================================================================

torch.signal.windows. hann ( *M*  , *** , *sym = True*  , *dtype = None*  , *layout = torch.strided*  , *device = None*  , *requires_grad = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/signal/windows/windows.py#L478) 
:   Computes the Hann window. 

The Hann window is defined as follows: 

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
<mfrac>
<mn>
             1
            </mn>
<mn>
             2
            </mn>
</mfrac>
<mtext>
</mtext>
<mrow>
<mo fence="true">
             [
            </mo>
<mn>
             1
            </mn>
<mo>
             −
            </mo>
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
<mo fence="true">
             ]
            </mo>
</mrow>
<mo>
            =
           </mo>
<msup>
<mrow>
<mi>
              sin
             </mi>
<mo>
              ⁡
             </mo>
</mrow>
<mn>
             2
            </mn>
</msup>
<mrow>
<mo fence="true">
             (
            </mo>
<mfrac>
<mrow>
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
           w_n = frac{1}{2} left[1 - cos left( frac{2 pi n}{M - 1} right)right] =
sin^2 left( frac{pi n}{M - 1} right)
          </annotation>
</semantics>
</math> -->
w n = 1 2 [ 1 − cos ⁡ ( 2 π n M − 1 ) ] = sin ⁡ 2 ( π n M − 1 ) w_n = frac{1}{2} left[1 - cos left( frac{2 pi n}{M - 1} right)right] =
sin^2 left( frac{pi n}{M - 1} right)

w n ​ = 2 1 ​ [ 1 − cos ( M − 1 2 πn ​ ) ] = sin 2 ( M − 1 πn ​ )

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
>>> # Generates a symmetric Hann window.
>>> torch.signal.windows.hann(10)
tensor([0.0000, 0.1170, 0.4132, 0.7500, 0.9698, 0.9698, 0.7500, 0.4132, 0.1170, 0.0000])

>>> # Generates a periodic Hann window.
>>> torch.signal.windows.hann(10, sym=False)
tensor([0.0000, 0.0955, 0.3455, 0.6545, 0.9045, 1.0000, 0.9045, 0.6545, 0.3455, 0.0955])

```

