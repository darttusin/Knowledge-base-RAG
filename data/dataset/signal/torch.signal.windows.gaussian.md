torch.signal.windows.gaussian 
==============================================================================================

torch.signal.windows. gaussian ( *M*  , *** , *std = 1.0*  , *sym = True*  , *dtype = None*  , *layout = torch.strided*  , *device = None*  , *requires_grad = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/signal/windows/windows.py#L256) 
:   Computes a window with a gaussian waveform. 

The gaussian window is defined as follows: 

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
            exp
           </mi>
<mo>
            ⁡
           </mo>
<mrow>
<mo fence="true">
             (
            </mo>
<mo>
             −
            </mo>
<msup>
<mrow>
<mo fence="true">
               (
              </mo>
<mfrac>
<mi>
                n
               </mi>
<mrow>
<mn>
                 2
                </mn>
<mi>
                 σ
                </mi>
</mrow>
</mfrac>
<mo fence="true">
               )
              </mo>
</mrow>
<mn>
              2
             </mn>
</msup>
<mo fence="true">
             )
            </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           w_n = exp{left(-left(frac{n}{2sigma}right)^2right)}
          </annotation>
</semantics>
</math> -->
w n = exp ⁡ ( − ( n 2 σ ) 2 ) w_n = exp{left(-left(frac{n}{2sigma}right)^2right)}

w n ​ = exp ( − ( 2 σ n ​ ) 2 )

The window is normalized to 1 (maximum value is 1). However, the 1 doesn’t appear if `M`  is even and `sym`  is *True* . 

Parameters
: **M** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the length of the window.
In other words, the number of points of the returned window.

Keyword Arguments
:   * **std** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – the standard deviation of the gaussian. It controls how narrow or wide the window is.
Default: 1.0.
* **sym** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If *False* , returns a periodic window suitable for use in spectral analysis.
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
>>> # Generates a symmetric gaussian window with a standard deviation of 1.0.
>>> torch.signal.windows.gaussian(10)
tensor([4.0065e-05, 2.1875e-03, 4.3937e-02, 3.2465e-01, 8.8250e-01, 8.8250e-01, 3.2465e-01, 4.3937e-02, 2.1875e-03, 4.0065e-05])

>>> # Generates a periodic gaussian window and standard deviation equal to 0.9.
>>> torch.signal.windows.gaussian(10, sym=False,std=0.9)
tensor([1.9858e-07, 5.1365e-05, 3.8659e-03, 8.4658e-02, 5.3941e-01, 1.0000e+00, 5.3941e-01, 8.4658e-02, 3.8659e-03, 5.1365e-05])

```

