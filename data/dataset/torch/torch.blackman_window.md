torch.blackman_window 
===============================================================================

torch. blackman_window ( *window_length*  , *periodic = True*  , *** , *dtype = None*  , *layout = torch.strided*  , *device = None*  , *requires_grad = False* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Blackman window function. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            w
           </mi>
<mo stretchy="false">
            [
           </mo>
<mi>
            n
           </mi>
<mo stretchy="false">
            ]
           </mo>
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
               N
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
               N
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
           w[n] = 0.42 - 0.5 cos left( frac{2 pi n}{N - 1} right) + 0.08 cos left( frac{4 pi n}{N - 1} right)
          </annotation>
</semantics>
</math> -->
w [ n ] = 0.42 − 0.5 cos ⁡ ( 2 π n N − 1 ) + 0.08 cos ⁡ ( 4 π n N − 1 ) w[n] = 0.42 - 0.5 cos left( frac{2 pi n}{N - 1} right) + 0.08 cos left( frac{4 pi n}{N - 1} right)

w [ n ] = 0.42 − 0.5 cos ( N − 1 2 πn ​ ) + 0.08 cos ( N − 1 4 πn ​ )

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            N
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           N
          </annotation>
</semantics>
</math> -->N NN  is the full window size. 

The input `window_length`  is a positive integer controlling the
returned window size. `periodic`  flag determines whether the returned
window trims off the last duplicate value from the symmetric window and is
ready to be used as a periodic window with functions like [`torch.stft()`](torch.stft.html#torch.stft "torch.stft")  . Therefore, if `periodic`  is true, the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            N
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           N
          </annotation>
</semantics>
</math> -->N NN  in
above formula is in fact <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            window_length
           </mtext>
<mo>
            +
           </mo>
<mn>
            1
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           text{window_length} + 1
          </annotation>
</semantics>
</math> -->window_length + 1 text{window_length} + 1window_length + 1  . Also, we always have `torch.blackman_window(L, periodic=True)`  equal to `torch.blackman_window(L + 1, periodic=False)[:-1]`  . 

Note 

If `window_length` <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
             =
            </mo>
<mn>
             1
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            =1
           </annotation>
</semantics>
</math> -->= 1 =1= 1  , the returned window contains a single value 1.

Parameters
:   * **window_length** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the size of returned window
* **periodic** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If True, returns a window to be used as periodic
function. If False, return a symmetric window.

Keyword Arguments
:   * **dtype** ( [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned tensor.
Default: if `None`  , uses a global default (see [`torch.set_default_dtype()`](torch.set_default_dtype.html#torch.set_default_dtype "torch.set_default_dtype")  ). Only floating point types are supported.
* **layout** ( [`torch.layout`](../tensor_attributes.html#torch.layout "torch.layout")  , optional) – the desired layout of returned window tensor. Only `torch.strided`  (dense layout) is supported.
* **device** ( [`torch.device`](../tensor_attributes.html#torch.device "torch.device")  , optional) – the desired device of returned tensor.
Default: if `None`  , uses the current device for the default tensor type
(see [`torch.set_default_device()`](torch.set_default_device.html#torch.set_default_device "torch.set_default_device")  ). [`device`](../tensor_attributes.html#torch.device "torch.device")  will be the CPU
for CPU tensor types and the current CUDA device for CUDA tensor types.
* **requires_grad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If autograd should record operations on the
returned tensor. Default: `False`  .

Returns
:   A 1-D tensor of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mtext>
              window_length
             </mtext>
<mo separator="true">
              ,
             </mo>
<mo stretchy="false">
              )
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             (text{window_length},)
            </annotation>
</semantics>
</math> -->( window_length , ) (text{window_length},)( window_length , )  containing the window

Return type
:   [Tensor](../tensors.html#torch.Tensor "torch.Tensor")

