torch.nn.functional.fractional_max_pool2d 
========================================================================================================================

torch.nn.functional. fractional_max_pool2d ( *input*  , *kernel_size*  , *output_size = None*  , *output_ratio = None*  , *return_indices = False*  , *_random_samples = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_jit_internal.py#L617) 
:   Applies 2D fractional max pooling over an input signal composed of several input planes. 

Fractional MaxPooling is described in detail in the paper [Fractional MaxPooling](http://arxiv.org/abs/1412.6071)  by Ben Graham 

The max-pooling operation is applied in <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            k
           </mi>
<mi>
            H
           </mi>
<mo>
            ×
           </mo>
<mi>
            k
           </mi>
<mi>
            W
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           kH times kW
          </annotation>
</semantics>
</math> -->k H × k W kH times kWk H × kW  regions by a stochastic
step size determined by the target output size.
The number of output features is equal to the number of input planes. 

Parameters
:   * **kernel_size** – the size of the window to take a max over.
Can be a single number <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                k
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               k
              </annotation>
</semantics>
</math> -->k kk  (for a square kernel of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                k
               </mi>
<mo>
                ×
               </mo>
<mi>
                k
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               k times k
              </annotation>
</semantics>
</math> -->k × k k times kk × k  )
or a tuple *(kH, kW)*

* **output_size** – the target output size of the image of the form <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                o
               </mi>
<mi>
                H
               </mi>
<mo>
                ×
               </mo>
<mi>
                o
               </mi>
<mi>
                W
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               oH times oW
              </annotation>
</semantics>
</math> -->o H × o W oH times oWoH × o W  .
Can be a tuple *(oH, oW)* or a single number <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                o
               </mi>
<mi>
                H
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               oH
              </annotation>
</semantics>
</math> -->o H oHoH  for a square image <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                o
               </mi>
<mi>
                H
               </mi>
<mo>
                ×
               </mo>
<mi>
                o
               </mi>
<mi>
                H
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               oH times oH
              </annotation>
</semantics>
</math> -->o H × o H oH times oHoH × oH

* **output_ratio** – If one wants to have an output size as a ratio of the input size, this option can be given.
This has to be a number or tuple in the range (0, 1)
* **return_indices** – if `True`  , will return the indices along with the outputs.
Useful to pass to [`max_unpool2d()`](torch.nn.functional.max_unpool2d.html#torch.nn.functional.max_unpool2d "torch.nn.functional.max_unpool2d")  .

Examples::
:   ```
>>> input = torch.randn(20, 16, 50, 32)
>>> # pool of square window of size=3, and target output size 13x12
>>> F.fractional_max_pool2d(input, 3, output_size=(13, 12))
>>> # pool of square window and target output size being half of input image size
>>> F.fractional_max_pool2d(input, 3, output_ratio=(0.5, 0.5))

```

