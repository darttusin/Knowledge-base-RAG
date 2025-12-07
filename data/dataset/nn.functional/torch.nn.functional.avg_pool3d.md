torch.nn.functional.avg_pool3d 
=================================================================================================

torch.nn.functional. avg_pool3d ( *input*  , *kernel_size*  , *stride = None*  , *padding = 0*  , *ceil_mode = False*  , *count_include_pad = True*  , *divisor_override = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Applies 3D average-pooling operation in <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            k
           </mi>
<mi>
            T
           </mi>
<mo>
            ×
           </mo>
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
           kT times kH times kW
          </annotation>
</semantics>
</math> -->k T × k H × k W kT times kH times kWk T × k H × kW  regions by step
size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            s
           </mi>
<mi>
            T
           </mi>
<mo>
            ×
           </mo>
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
           sT times sH times sW
          </annotation>
</semantics>
</math> -->s T × s H × s W sT times sH times sWs T × sH × s W  steps. The number of output features is equal to <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            ⌊
           </mo>
<mfrac>
<mtext>
             input planes
            </mtext>
<mrow>
<mi>
              s
             </mi>
<mi>
              T
             </mi>
</mrow>
</mfrac>
<mo stretchy="false">
            ⌋
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           lfloorfrac{text{input planes}}{sT}rfloor
          </annotation>
</semantics>
</math> -->⌊ input planes s T ⌋ lfloorfrac{text{input planes}}{sT}rfloor⌊ s T input planes ​ ⌋  . 

Note 

pad should be at most half of effective kernel size.

See [`AvgPool3d`](torch.nn.AvgPool3d.html#torch.nn.AvgPool3d "torch.nn.AvgPool3d")  for details and output shape. 

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
                T
               </mi>
<mo>
                ×
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
               (text{minibatch} , text{in_channels} , iT times iH , iW)
              </annotation>
</semantics>
</math> -->( minibatch , in_channels , i T × i H , i W ) (text{minibatch} , text{in_channels} , iT times iH , iW)( minibatch , in_channels , i T × i H , iW )

* **kernel_size** – size of the pooling region. Can be a single number or a
tuple *(kT, kH, kW)*
* **stride** – stride of the pooling operation. Can be a single number or a
tuple *(sT, sH, sW)* . Default: `kernel_size`
* **padding** – implicit zero paddings on both sides of the input. Can be a
single number or a tuple *(padT, padH, padW)* , Default: 0
* **ceil_mode** – when True, will use *ceil* instead of *floor* in the formula
to compute the output shape
* **count_include_pad** – when True, will include the zero-padding in the
averaging calculation
* **divisor_override** – if specified, it will be used as divisor, otherwise
size of the pooling region will be used. Default: None

