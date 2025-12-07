torch.hamming_window 
=============================================================================

torch. hamming_window ( *window_length*  , *periodic = True*  , *alpha = 0.54*  , *beta = 0.46*  , *** , *dtype = None*  , *layout = torch.strided*  , *device = None*  , *requires_grad = False* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Hamming window function. 

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
<mo separator="true">
            ,
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           w[n] = alpha - beta cos left( frac{2 pi n}{N - 1} right),
          </annotation>
</semantics>
</math> -->
w [ n ] = α − β cos ⁡ ( 2 π n N − 1 ) , w[n] = alpha - beta cos left( frac{2 pi n}{N - 1} right),

w [ n ] = α − β cos ( N − 1 2 πn ​ ) ,

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
</math> -->window_length + 1 text{window_length} + 1window_length + 1  . Also, we always have `torch.hamming_window(L, periodic=True)`  equal to `torch.hamming_window(L + 1, periodic=False)[:-1])`  . 

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

Note 

This is a generalized version of [`torch.hann_window()`](torch.hann_window.html#torch.hann_window "torch.hann_window")  .

Parameters
:   * **window_length** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the size of returned window
* **periodic** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If True, returns a window to be used as periodic
function. If False, return a symmetric window.
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
</math> -->α alphaα  in the equation above

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
</math> -->β betaβ  in the equation above

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
</math> -->( window_length , ) (text{window_length},)( window_length , )  containing the window.

Return type
:   [Tensor](../tensors.html#torch.Tensor "torch.Tensor")

