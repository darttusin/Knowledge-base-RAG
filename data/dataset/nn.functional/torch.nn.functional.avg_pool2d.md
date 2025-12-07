torch.nn.functional.avg_pool2d 
=================================================================================================

torch.nn.functional. avg_pool2d ( *input*  , *kernel_size*  , *stride = None*  , *padding = 0*  , *ceil_mode = False*  , *count_include_pad = True*  , *divisor_override = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Applies 2D average-pooling operation in <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->k H × k W kH times kWk H × kW  regions by step size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            s
           </mi>
<mi>
            H
           </mi>
<mo>
            ×
           </mo>
<mi>
            s
           </mi>
<mi>
            W
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           sH times sW
          </annotation>
</semantics>
</math> -->s H × s W sH times sWsH × s W  steps. The number of output features is equal to the number of
input planes. 

Note 

pad should be at most half of effective kernel size.

See [`AvgPool2d`](torch.nn.AvgPool2d.html#torch.nn.AvgPool2d "torch.nn.AvgPool2d")  for details and output shape. 

Parameters
:   * **input** – input tensor <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mtext>
                minibatch
               </mtext>
<mo separator="true">
                ,
               </mo>
<mtext>
                in_channels
               </mtext>
<mo separator="true">
                ,
               </mo>
<mi>
                i
               </mi>
<mi>
                H
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                i
               </mi>
<mi>
                W
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (text{minibatch} , text{in_channels} , iH , iW)
              </annotation>
</semantics>
</math> -->( minibatch , in_channels , i H , i W ) (text{minibatch} , text{in_channels} , iH , iW)( minibatch , in_channels , i H , iW )

* **kernel_size** – size of the pooling region. Can be a single number, a single-element tuple or a
tuple *(kH, kW)*
* **stride** – stride of the pooling operation. Can be a single number, a single-element tuple or a
tuple *(sH, sW)* . Default: `kernel_size`
* **padding** – implicit zero paddings on both sides of the input. Can be a
single number, a single-element tuple or a tuple *(padH, padW)* . Default: 0
* **ceil_mode** – when True, will use *ceil* instead of *floor* in the formula
to compute the output shape. Default: `False`
* **count_include_pad** – when True, will include the zero-padding in the
averaging calculation. Default: `True`
* **divisor_override** – if specified, it will be used as divisor, otherwise
size of the pooling region will be used. Default: None

