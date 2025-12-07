torch.kaiser_window 
===========================================================================

torch. kaiser_window ( *window_length*  , *periodic = True*  , *beta = 12.0*  , *** , *dtype = None*  , *layout = torch.strided*  , *device = None*  , *requires_grad = False* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the Kaiser window with window length `window_length`  and shape parameter `beta`  . 

Let I_0 be the zeroth order modified Bessel function of the first kind (see [`torch.i0()`](torch.i0.html#torch.i0 "torch.i0")  ) and `N = L - 1`  if `periodic`  is False and `L`  if `periodic`  is True,
where `L`  is the `window_length`  . This function computes: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            o
           </mi>
<mi>
            u
           </mi>
<msub>
<mi>
             t
            </mi>
<mi>
             i
            </mi>
</msub>
<mo>
            =
           </mo>
<msub>
<mi>
             I
            </mi>
<mn>
             0
            </mn>
</msub>
<mrow>
<mo fence="true">
             (
            </mo>
<mi>
             β
            </mi>
<msqrt>
<mrow>
<mn>
               1
              </mn>
<mo>
               −
              </mo>
<msup>
<mrow>
<mo fence="true">
                 (
                </mo>
<mfrac>
<mrow>
<mi>
                   i
                  </mi>
<mo>
                   −
                  </mo>
<mi>
                   N
                  </mi>
<mi mathvariant="normal">
                   /
                  </mi>
<mn>
                   2
                  </mn>
</mrow>
<mrow>
<mi>
                   N
                  </mi>
<mi mathvariant="normal">
                   /
                  </mi>
<mn>
                   2
                  </mn>
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
</mrow>
</msqrt>
<mo fence="true">
             )
            </mo>
</mrow>
<mi mathvariant="normal">
            /
           </mi>
<msub>
<mi>
             I
            </mi>
<mn>
             0
            </mn>
</msub>
<mo stretchy="false">
            (
           </mo>
<mi>
            β
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           out_i = I_0 left( beta sqrt{1 - left( {frac{i - N/2}{N/2}} right) ^2 } right) / I_0( beta )
          </annotation>
</semantics>
</math> -->
o u t i = I 0 ( β 1 − ( i − N / 2 N / 2 ) 2 ) / I 0 ( β ) out_i = I_0 left( beta sqrt{1 - left( {frac{i - N/2}{N/2}} right) ^2 } right) / I_0( beta )

o u t i ​ = I 0 ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjMuNjAwZW0iIHZpZXdib3g9IjAgMCA4NzUgMzYwMCIgd2lkdGg9IjAuODc1ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik04NjMsOWMwLC0yLC0yLC01LC02LC05YzAsMCwtMTcsMCwtMTcsMGMtMTIuNywwLC0xOS4zLDAuMywtMjAsMQpjLTUuMyw1LjMsLTEwLjMsMTEsLTE1LDE3Yy0yNDIuNywyOTQuNywtMzk1LjMsNjgyLC00NTgsMTE2MmMtMjEuMywxNjMuMywtMzMuMywzNDksCi0zNiw1NTcgbDAsODRjMC4yLDYsMCwyNiwwLDYwYzIsMTU5LjMsMTAsMzEwLjcsMjQsNDU0YzUzLjMsNTI4LDIxMCwKOTQ5LjcsNDcwLDEyNjVjNC43LDYsOS43LDExLjcsMTUsMTdjMC43LDAuNyw3LDEsMTksMWMwLDAsMTgsMCwxOCwwYzQsLTQsNiwtNyw2LC05CmMwLC0yLjcsLTMuMywtOC43LC0xMCwtMThjLTEzNS4zLC0xOTIuNywtMjM1LjUsLTQxNC4zLC0zMDAuNSwtNjY1Yy02NSwtMjUwLjcsLTEwMi41LAotNTQ0LjcsLTExMi41LC04ODJjLTIsLTEwNCwtMywtMTY3LC0zLC0xODkKbDAsLTkyYzAsLTE2Mi43LDUuNywtMzE0LDE3LC00NTRjMjAuNywtMjcyLDYzLjcsLTUxMywxMjksLTcyM2M2NS4zLAotMjEwLDE1NS4zLC0zOTYuMywyNzAsLTU1OWM2LjcsLTkuMywxMCwtMTUuMywxMCwtMTh6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ β 1 − ( N /2 i − N /2 ​ ) 2 ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjMuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMzI0MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNNDczLDI3OTMKYzMzOS4zLC0xNzk5LjMsNTA5LjMsLTI3MDAsNTEwLC0yNzAyIGwwIC0wCmMzLjMsLTcuMyw5LjMsLTExLDE4LC0xMSBINDAwMDAwdjQwSDEwMTcuNwpzLTkwLjUsNDc4LC0yNzYuMiwxNDY2Yy0xODUuNyw5ODgsLTI3OS41LDE0ODMsLTI4MS41LDE0ODVjLTIsNiwtMTAsOSwtMjQsOQpjLTgsMCwtMTIsLTAuNywtMTIsLTJjMCwtMS4zLC01LjMsLTMyLC0xNiwtOTJjLTUwLjcsLTI5My4zLC0xMTkuNywtNjkzLjMsLTIwNywtMTIwMApjMCwtMS4zLC01LjMsOC43LC0xNiwzMGMtMTAuNywyMS4zLC0yMS4zLDQyLjcsLTMyLDY0cy0xNiwzMywtMTYsMzNzLTI2LC0yNiwtMjYsLTI2CnM3NiwtMTUzLDc2LC0xNTNzNzcsLTE1MSw3NywtMTUxYzAuNywwLjcsMzUuNywyMDIsMTA1LDYwNGM2Ny4zLDQwMC43LDEwMiw2MDIuNywxMDQsCjYwNnpNMTAwMSA4MGg0MDAwMDB2NDBIMTAxNy43eiI+CjwvcGF0aD4KPC9zdmc+)​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjMuNjAwZW0iIHZpZXdib3g9IjAgMCA4NzUgMzYwMCIgd2lkdGg9IjAuODc1ZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik03NiwwYy0xNi43LDAsLTI1LDMsLTI1LDljMCwyLDIsNi4zLDYsMTNjMjEuMywyOC43LDQyLjMsNjAuMywKNjMsOTVjOTYuNywxNTYuNywxNzIuOCwzMzIuNSwyMjguNSw1MjcuNWM1NS43LDE5NSw5Mi44LDQxNi41LDExMS41LDY2NC41CmMxMS4zLDEzOS4zLDE3LDI5MC43LDE3LDQ1NGMwLDI4LDEuNyw0MywzLjMsNDVsMCw5CmMtMyw0LC0zLjMsMTYuNywtMy4zLDM4YzAsMTYyLC01LjcsMzEzLjcsLTE3LDQ1NWMtMTguNywyNDgsLTU1LjgsNDY5LjMsLTExMS41LDY2NApjLTU1LjcsMTk0LjcsLTEzMS44LDM3MC4zLC0yMjguNSw1MjdjLTIwLjcsMzQuNywtNDEuNyw2Ni4zLC02Myw5NWMtMiwzLjMsLTQsNywtNiwxMQpjMCw3LjMsNS43LDExLDE3LDExYzAsMCwxMSwwLDExLDBjOS4zLDAsMTQuMywtMC4zLDE1LC0xYzUuMywtNS4zLDEwLjMsLTExLDE1LC0xNwpjMjQyLjcsLTI5NC43LDM5NS4zLC02ODEuNyw0NTgsLTExNjFjMjEuMywtMTY0LjcsMzMuMywtMzUwLjcsMzYsLTU1OApsMCwtMTQ0Yy0yLC0xNTkuMywtMTAsLTMxMC43LC0yNCwtNDU0Yy01My4zLC01MjgsLTIxMCwtOTQ5LjcsCi00NzAsLTEyNjVjLTQuNywtNiwtOS43LC0xMS43LC0xNSwtMTdjLTAuNywtMC43LC02LjcsLTEsLTE4LC0xeiI+CjwvcGF0aD4KPC9zdmc+)​ / I 0 ​ ( β )

Calling `torch.kaiser_window(L, B, periodic=True)`  is equivalent to calling `torch.kaiser_window(L + 1, B, periodic=False)[:-1])`  .
The `periodic`  argument is intended as a helpful shorthand
to produce a periodic window as input to functions like [`torch.stft()`](torch.stft.html#torch.stft "torch.stft")  . 

Note 

If `window_length`  is one, then the returned window is a single element tensor containing a one.

Parameters
:   * **window_length** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – length of the window.
* **periodic** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If True, returns a periodic window suitable for use in spectral analysis.
If False, returns a symmetric window suitable for use in filter design.
* **beta** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – shape parameter for the window.

Keyword Arguments
:   * **dtype** ( [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned tensor.
Default: if `None`  , uses a global default (see [`torch.set_default_dtype()`](torch.set_default_dtype.html#torch.set_default_dtype "torch.set_default_dtype")  ).
* **layout** ( [`torch.layout`](../tensor_attributes.html#torch.layout "torch.layout")  , optional) – the desired layout of returned window tensor. Only `torch.strided`  (dense layout) is supported.
* **device** ( [`torch.device`](../tensor_attributes.html#torch.device "torch.device")  , optional) – the desired device of returned tensor.
Default: if `None`  , uses the current device for the default tensor type
(see [`torch.set_default_device()`](torch.set_default_device.html#torch.set_default_device "torch.set_default_device")  ). [`device`](../tensor_attributes.html#torch.device "torch.device")  will be the CPU
for CPU tensor types and the current CUDA device for CUDA tensor types.
* **requires_grad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If autograd should record operations on the
returned tensor. Default: `False`  .

