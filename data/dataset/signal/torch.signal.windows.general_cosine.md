torch.signal.windows.general_cosine 
===========================================================================================================

torch.signal.windows. general_cosine ( *M*  , *** , *a*  , *sym = True*  , *dtype = None*  , *layout = torch.strided*  , *device = None*  , *requires_grad = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/signal/windows/windows.py#L674) 
:   Computes the general cosine window. 

The general cosine window is defined as follows: 

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
<munderover>
<mo>
             ∑
            </mo>
<mrow>
<mi>
              i
             </mi>
<mo>
              =
             </mo>
<mn>
              0
             </mn>
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
</munderover>
<mo stretchy="false">
            (
           </mo>
<mo>
            −
           </mo>
<mn>
            1
           </mn>
<msup>
<mo stretchy="false">
             )
            </mo>
<mi>
             i
            </mi>
</msup>
<msub>
<mi>
             a
            </mi>
<mi>
             i
            </mi>
</msub>
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
               i
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
           w_n = sum^{M-1}_{i=0} (-1)^i a_i cos{ left( frac{2 pi i n}{M - 1}right)}
          </annotation>
</semantics>
</math> -->
w n = ∑ i = 0 M − 1 ( − 1 ) i a i cos ⁡ ( 2 π i n M − 1 ) w_n = sum^{M-1}_{i=0} (-1)^i a_i cos{ left( frac{2 pi i n}{M - 1}right)}

w n ​ = i = 0 ∑ M − 1 ​ ( − 1 ) i a i ​ cos ( M − 1 2 πin ​ )

The window is normalized to 1 (maximum value is 1). However, the 1 doesn’t appear if `M`  is even and `sym`  is *True* . 

Parameters
: **M** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the length of the window.
In other words, the number of points of the returned window.

Keyword Arguments
:   * **a** ( *Iterable*  ) – the coefficients associated to each of the cosine functions.
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
>>> # Generates a symmetric general cosine window with 3 coefficients.
>>> torch.signal.windows.general_cosine(10, a=[0.46, 0.23, 0.31], sym=True)
tensor([0.5400, 0.3376, 0.1288, 0.4200, 0.9136, 0.9136, 0.4200, 0.1288, 0.3376, 0.5400])

>>> # Generates a periodic general cosine window with 2 coefficients.
>>> torch.signal.windows.general_cosine(10, a=[0.5, 1 - 0.5], sym=False)
tensor([0.0000, 0.0955, 0.3455, 0.6545, 0.9045, 1.0000, 0.9045, 0.6545, 0.3455, 0.0955])

```

