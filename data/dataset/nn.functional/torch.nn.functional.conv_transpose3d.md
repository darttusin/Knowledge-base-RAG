torch.nn.functional.conv_transpose3d 
=============================================================================================================

torch.nn.functional. conv_transpose3d ( *input*  , *weight*  , *bias = None*  , *stride = 1*  , *padding = 0*  , *output_padding = 0*  , *groups = 1*  , *dilation = 1* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Applies a 3D transposed convolution operator over an input image
composed of several input planes, sometimes also called “deconvolution” 

This operator supports [TensorFloat32](../notes/cuda.html#tf32-on-ampere)  . 

See [`ConvTranspose3d`](torch.nn.ConvTranspose3d.html#torch.nn.ConvTranspose3d "torch.nn.ConvTranspose3d")  for details and output shape. 

Note 

In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting `torch.backends.cudnn.deterministic = True`  . See [Reproducibility](../notes/randomness.html)  for more information.

Parameters
:   * **input** – input tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
               (text{minibatch} , text{in_channels} , iT , iH , iW)
              </annotation>
</semantics>
</math> -->( minibatch , in_channels , i T , i H , i W ) (text{minibatch} , text{in_channels} , iT , iH , iW)( minibatch , in_channels , i T , i H , iW )

* **weight** – filters of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mtext>
                in_channels
               </mtext>
<mo separator="true">
                ,
               </mo>
<mfrac>
<mtext>
                 out_channels
                </mtext>
<mtext>
                 groups
                </mtext>
</mfrac>
<mo separator="true">
                ,
               </mo>
<mi>
                k
               </mi>
<mi>
                T
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                k
               </mi>
<mi>
                H
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                k
               </mi>
<mi>
                W
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (text{in_channels} , frac{text{out_channels}}{text{groups}} , kT , kH , kW)
              </annotation>
</semantics>
</math> -->( in_channels , out_channels groups , k T , k H , k W ) (text{in_channels} , frac{text{out_channels}}{text{groups}} , kT , kH , kW)( in_channels , groups out_channels ​ , k T , k H , kW )

* **bias** – optional bias of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mtext>
                out_channels
               </mtext>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (text{out_channels})
              </annotation>
</semantics>
</math> -->( out_channels ) (text{out_channels})( out_channels )  . Default: None

* **stride** – the stride of the convolving kernel. Can be a single number or a
tuple `(sT, sH, sW)`  . Default: 1
* **padding** – `dilation * (kernel_size - 1) - padding`  zero-padding will be added to both
sides of each dimension in the input. Can be a single number or a tuple `(padT, padH, padW)`  . Default: 0
* **output_padding** – additional size added to one side of each dimension in the
output shape. Can be a single number or a tuple `(out_padT, out_padH, out_padW)`  . Default: 0
* **groups** – split input into groups, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
                in_channels
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               text{in_channels}
              </annotation>
</semantics>
</math> -->in_channels text{in_channels}in_channels  should be divisible by the
number of groups. Default: 1

* **dilation** – the spacing between kernel elements. Can be a single number or
a tuple *(dT, dH, dW)* . Default: 1

Examples: 

```
>>> inputs = torch.randn(20, 16, 50, 10, 20)
>>> weights = torch.randn(16, 33, 3, 3, 3)
>>> F.conv_transpose3d(inputs, weights)

```

