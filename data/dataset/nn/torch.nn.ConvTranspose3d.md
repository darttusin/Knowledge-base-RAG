ConvTranspose3d 
==================================================================

*class* torch.nn. ConvTranspose3d ( *in_channels*  , *out_channels*  , *kernel_size*  , *stride = 1*  , *padding = 0*  , *output_padding = 0*  , *groups = 1*  , *bias = True*  , *dilation = 1*  , *padding_mode = 'zeros'*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/conv.py#L1173) 
:   Applies a 3D transposed convolution operator over an input image composed of several input
planes.
The transposed convolution operator multiplies each input value element-wise by a learnable kernel,
and sums over the outputs from all input feature planes. 

This module can be seen as the gradient of Conv3d with respect to its input.
It is also known as a fractionally-strided convolution or
a deconvolution (although it is not an actual deconvolution operation as it does
not compute a true inverse of convolution). For more information, see the visualizations [here](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)  and the [Deconvolutional Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)  paper. 

This module supports [TensorFloat32](../notes/cuda.html#tf32-on-ampere)  . 

On certain ROCm devices, when using float16 inputs this module will use [different precision](../notes/numerical_accuracy.html#fp16-on-mi200)  for backward. 

* `stride`  controls the stride for the cross-correlation.
* `padding`  controls the amount of implicit zero padding on both
sides for `dilation * (kernel_size - 1) - padding`  number of points. See note
below for details.
* `output_padding`  controls the additional size added to one side
of the output shape. See note below for details.
* `dilation`  controls the spacing between the kernel points; also known as the à trous algorithm.
It is harder to describe, but the link [here](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)  has a nice visualization of what `dilation`  does.
* `groups`  controls the connections between inputs and outputs. `in_channels`  and `out_channels`  must both be divisible by `groups`  . For example,

    > + At groups=1, all inputs are convolved to all outputs.
        > + At groups=2, the operation becomes equivalent to having two conv
        > layers side by side, each seeing half the input channels
        > and producing half the output channels, and both subsequently
        > concatenated.
        > + At groups= `in_channels`  , each input channel is convolved with
        > its own set of filters (of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
        > <semantics>
        > <mrow>
        > <mfrac>
        > <mtext>
        > out_channels
        > </mtext>
        > <mtext>
        > in_channels
        > </mtext>
        > </mfrac>
        > </mrow>
        > <annotation encoding="application/x-tex">
        > frac{text{out_channels}}{text{in_channels}}
        > </annotation>
        > </semantics>
        > </math> -->out_channels in_channels frac{text{out_channels}}{text{in_channels}}in_channels out_channels ​  ).

The parameters `kernel_size`  , `stride`  , `padding`  , `output_padding`  can either be: 

> * a single `int`  – in which case the same value is used for the depth, height and width dimensions
> * a `tuple`  of three ints – in which case, the first *int* is used for the depth dimension,
> the second *int* for the height dimension and the third *int* for the width dimension

Note 

The `padding`  argument effectively adds `dilation * (kernel_size - 1) - padding`  amount of zero padding to both sizes of the input. This is set so that
when a [`Conv3d`](torch.nn.Conv3d.html#torch.nn.Conv3d "torch.nn.Conv3d")  and a [`ConvTranspose3d`](#torch.nn.ConvTranspose3d "torch.nn.ConvTranspose3d")  are initialized with same parameters, they are inverses of each other in
regard to the input and output shapes. However, when `stride > 1`  , [`Conv3d`](torch.nn.Conv3d.html#torch.nn.Conv3d "torch.nn.Conv3d")  maps multiple input shapes to the same output
shape. `output_padding`  is provided to resolve this ambiguity by
effectively increasing the calculated output shape on one side. Note
that `output_padding`  is only used to find output shape, but does
not actually add zero-padding to output.

Note 

In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting `torch.backends.cudnn.deterministic = True`  . See [Reproducibility](../notes/randomness.html)  for more information.

Parameters
:   * **in_channels** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Number of channels in the input image
* **out_channels** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Number of channels produced by the convolution
* **kernel_size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  ) – Size of the convolving kernel
* **stride** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – Stride of the convolution. Default: 1
* **padding** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – `dilation * (kernel_size - 1) - padding`  zero-padding
will be added to both sides of each dimension in the input. Default: 0
* **output_padding** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – Additional size added to one side
of each dimension in the output shape. Default: 0
* **groups** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – Number of blocked connections from input channels to output channels. Default: 1
* **bias** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If `True`  , adds a learnable bias to the output. Default: `True`
* **dilation** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – Spacing between kernel elements. Default: 1

Shape:
:   * Input: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 C
                </mi>
<mrow>
<mi>
                  i
                 </mi>
<mi>
                  n
                 </mi>
</mrow>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 D
                </mi>
<mrow>
<mi>
                  i
                 </mi>
<mi>
                  n
                 </mi>
</mrow>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 H
                </mi>
<mrow>
<mi>
                  i
                 </mi>
<mi>
                  n
                 </mi>
</mrow>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 W
                </mi>
<mrow>
<mi>
                  i
                 </mi>
<mi>
                  n
                 </mi>
</mrow>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C_{in}, D_{in}, H_{in}, W_{in})
              </annotation>
</semantics>
</math> -->( N , C i n , D i n , H i n , W i n ) (N, C_{in}, D_{in}, H_{in}, W_{in})( N , C in ​ , D in ​ , H in ​ , W in ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 C
                </mi>
<mrow>
<mi>
                  i
                 </mi>
<mi>
                  n
                 </mi>
</mrow>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 D
                </mi>
<mrow>
<mi>
                  i
                 </mi>
<mi>
                  n
                 </mi>
</mrow>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 H
                </mi>
<mrow>
<mi>
                  i
                 </mi>
<mi>
                  n
                 </mi>
</mrow>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 W
                </mi>
<mrow>
<mi>
                  i
                 </mi>
<mi>
                  n
                 </mi>
</mrow>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (C_{in}, D_{in}, H_{in}, W_{in})
              </annotation>
</semantics>
</math> -->( C i n , D i n , H i n , W i n ) (C_{in}, D_{in}, H_{in}, W_{in})( C in ​ , D in ​ , H in ​ , W in ​ )

* Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 C
                </mi>
<mrow>
<mi>
                  o
                 </mi>
<mi>
                  u
                 </mi>
<mi>
                  t
                 </mi>
</mrow>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 D
                </mi>
<mrow>
<mi>
                  o
                 </mi>
<mi>
                  u
                 </mi>
<mi>
                  t
                 </mi>
</mrow>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 H
                </mi>
<mrow>
<mi>
                  o
                 </mi>
<mi>
                  u
                 </mi>
<mi>
                  t
                 </mi>
</mrow>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 W
                </mi>
<mrow>
<mi>
                  o
                 </mi>
<mi>
                  u
                 </mi>
<mi>
                  t
                 </mi>
</mrow>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C_{out}, D_{out}, H_{out}, W_{out})
              </annotation>
</semantics>
</math> -->( N , C o u t , D o u t , H o u t , W o u t ) (N, C_{out}, D_{out}, H_{out}, W_{out})( N , C o u t ​ , D o u t ​ , H o u t ​ , W o u t ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 C
                </mi>
<mrow>
<mi>
                  o
                 </mi>
<mi>
                  u
                 </mi>
<mi>
                  t
                 </mi>
</mrow>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 D
                </mi>
<mrow>
<mi>
                  o
                 </mi>
<mi>
                  u
                 </mi>
<mi>
                  t
                 </mi>
</mrow>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 H
                </mi>
<mrow>
<mi>
                  o
                 </mi>
<mi>
                  u
                 </mi>
<mi>
                  t
                 </mi>
</mrow>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 W
                </mi>
<mrow>
<mi>
                  o
                 </mi>
<mi>
                  u
                 </mi>
<mi>
                  t
                 </mi>
</mrow>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (C_{out}, D_{out}, H_{out}, W_{out})
              </annotation>
</semantics>
</math> -->( C o u t , D o u t , H o u t , W o u t ) (C_{out}, D_{out}, H_{out}, W_{out})( C o u t ​ , D o u t ​ , H o u t ​ , W o u t ​ )  , where

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
               D
              </mi>
<mrow>
<mi>
                o
               </mi>
<mi>
                u
               </mi>
<mi>
                t
               </mi>
</mrow>
</msub>
<mo>
              =
             </mo>
<mo stretchy="false">
              (
             </mo>
<msub>
<mi>
               D
              </mi>
<mrow>
<mi>
                i
               </mi>
<mi>
                n
               </mi>
</mrow>
</msub>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
<mo stretchy="false">
              )
             </mo>
<mo>
              ×
             </mo>
<mtext>
              stride
             </mtext>
<mo stretchy="false">
              [
             </mo>
<mn>
              0
             </mn>
<mo stretchy="false">
              ]
             </mo>
<mo>
              −
             </mo>
<mn>
              2
             </mn>
<mo>
              ×
             </mo>
<mtext>
              padding
             </mtext>
<mo stretchy="false">
              [
             </mo>
<mn>
              0
             </mn>
<mo stretchy="false">
              ]
             </mo>
<mo>
              +
             </mo>
<mtext>
              dilation
             </mtext>
<mo stretchy="false">
              [
             </mo>
<mn>
              0
             </mn>
<mo stretchy="false">
              ]
             </mo>
<mo>
              ×
             </mo>
<mo stretchy="false">
              (
             </mo>
<mtext>
              kernel_size
             </mtext>
<mo stretchy="false">
              [
             </mo>
<mn>
              0
             </mn>
<mo stretchy="false">
              ]
             </mo>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
<mo stretchy="false">
              )
             </mo>
<mo>
              +
             </mo>
<mtext>
              output_padding
             </mtext>
<mo stretchy="false">
              [
             </mo>
<mn>
              0
             </mn>
<mo stretchy="false">
              ]
             </mo>
<mo>
              +
             </mo>
<mn>
              1
             </mn>
</mrow>
<annotation encoding="application/x-tex">
             D_{out} = (D_{in} - 1) times text{stride}[0] - 2 times text{padding}[0] + text{dilation}[0]
          times (text{kernel_size}[0] - 1) + text{output_padding}[0] + 1
            </annotation>
</semantics>
</math> -->
D o u t = ( D i n − 1 ) × stride [ 0 ] − 2 × padding [ 0 ] + dilation [ 0 ] × ( kernel_size [ 0 ] − 1 ) + output_padding [ 0 ] + 1 D_{out} = (D_{in} - 1) times text{stride}[0] - 2 times text{padding}[0] + text{dilation}[0]
 times (text{kernel_size}[0] - 1) + text{output_padding}[0] + 1

D o u t ​ = ( D in ​ − 1 ) × stride [ 0 ] − 2 × padding [ 0 ] + dilation [ 0 ] × ( kernel_size [ 0 ] − 1 ) + output_padding [ 0 ] + 1

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
               H
              </mi>
<mrow>
<mi>
                o
               </mi>
<mi>
                u
               </mi>
<mi>
                t
               </mi>
</mrow>
</msub>
<mo>
              =
             </mo>
<mo stretchy="false">
              (
             </mo>
<msub>
<mi>
               H
              </mi>
<mrow>
<mi>
                i
               </mi>
<mi>
                n
               </mi>
</mrow>
</msub>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
<mo stretchy="false">
              )
             </mo>
<mo>
              ×
             </mo>
<mtext>
              stride
             </mtext>
<mo stretchy="false">
              [
             </mo>
<mn>
              1
             </mn>
<mo stretchy="false">
              ]
             </mo>
<mo>
              −
             </mo>
<mn>
              2
             </mn>
<mo>
              ×
             </mo>
<mtext>
              padding
             </mtext>
<mo stretchy="false">
              [
             </mo>
<mn>
              1
             </mn>
<mo stretchy="false">
              ]
             </mo>
<mo>
              +
             </mo>
<mtext>
              dilation
             </mtext>
<mo stretchy="false">
              [
             </mo>
<mn>
              1
             </mn>
<mo stretchy="false">
              ]
             </mo>
<mo>
              ×
             </mo>
<mo stretchy="false">
              (
             </mo>
<mtext>
              kernel_size
             </mtext>
<mo stretchy="false">
              [
             </mo>
<mn>
              1
             </mn>
<mo stretchy="false">
              ]
             </mo>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
<mo stretchy="false">
              )
             </mo>
<mo>
              +
             </mo>
<mtext>
              output_padding
             </mtext>
<mo stretchy="false">
              [
             </mo>
<mn>
              1
             </mn>
<mo stretchy="false">
              ]
             </mo>
<mo>
              +
             </mo>
<mn>
              1
             </mn>
</mrow>
<annotation encoding="application/x-tex">
             H_{out} = (H_{in} - 1) times text{stride}[1] - 2 times text{padding}[1] + text{dilation}[1]
          times (text{kernel_size}[1] - 1) + text{output_padding}[1] + 1
            </annotation>
</semantics>
</math> -->
H o u t = ( H i n − 1 ) × stride [ 1 ] − 2 × padding [ 1 ] + dilation [ 1 ] × ( kernel_size [ 1 ] − 1 ) + output_padding [ 1 ] + 1 H_{out} = (H_{in} - 1) times text{stride}[1] - 2 times text{padding}[1] + text{dilation}[1]
 times (text{kernel_size}[1] - 1) + text{output_padding}[1] + 1

H o u t ​ = ( H in ​ − 1 ) × stride [ 1 ] − 2 × padding [ 1 ] + dilation [ 1 ] × ( kernel_size [ 1 ] − 1 ) + output_padding [ 1 ] + 1

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
               W
              </mi>
<mrow>
<mi>
                o
               </mi>
<mi>
                u
               </mi>
<mi>
                t
               </mi>
</mrow>
</msub>
<mo>
              =
             </mo>
<mo stretchy="false">
              (
             </mo>
<msub>
<mi>
               W
              </mi>
<mrow>
<mi>
                i
               </mi>
<mi>
                n
               </mi>
</mrow>
</msub>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
<mo stretchy="false">
              )
             </mo>
<mo>
              ×
             </mo>
<mtext>
              stride
             </mtext>
<mo stretchy="false">
              [
             </mo>
<mn>
              2
             </mn>
<mo stretchy="false">
              ]
             </mo>
<mo>
              −
             </mo>
<mn>
              2
             </mn>
<mo>
              ×
             </mo>
<mtext>
              padding
             </mtext>
<mo stretchy="false">
              [
             </mo>
<mn>
              2
             </mn>
<mo stretchy="false">
              ]
             </mo>
<mo>
              +
             </mo>
<mtext>
              dilation
             </mtext>
<mo stretchy="false">
              [
             </mo>
<mn>
              2
             </mn>
<mo stretchy="false">
              ]
             </mo>
<mo>
              ×
             </mo>
<mo stretchy="false">
              (
             </mo>
<mtext>
              kernel_size
             </mtext>
<mo stretchy="false">
              [
             </mo>
<mn>
              2
             </mn>
<mo stretchy="false">
              ]
             </mo>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
<mo stretchy="false">
              )
             </mo>
<mo>
              +
             </mo>
<mtext>
              output_padding
             </mtext>
<mo stretchy="false">
              [
             </mo>
<mn>
              2
             </mn>
<mo stretchy="false">
              ]
             </mo>
<mo>
              +
             </mo>
<mn>
              1
             </mn>
</mrow>
<annotation encoding="application/x-tex">
             W_{out} = (W_{in} - 1) times text{stride}[2] - 2 times text{padding}[2] + text{dilation}[2]
          times (text{kernel_size}[2] - 1) + text{output_padding}[2] + 1
            </annotation>
</semantics>
</math> -->
W o u t = ( W i n − 1 ) × stride [ 2 ] − 2 × padding [ 2 ] + dilation [ 2 ] × ( kernel_size [ 2 ] − 1 ) + output_padding [ 2 ] + 1 W_{out} = (W_{in} - 1) times text{stride}[2] - 2 times text{padding}[2] + text{dilation}[2]
 times (text{kernel_size}[2] - 1) + text{output_padding}[2] + 1

W o u t ​ = ( W in ​ − 1 ) × stride [ 2 ] − 2 × padding [ 2 ] + dilation [ 2 ] × ( kernel_size [ 2 ] − 1 ) + output_padding [ 2 ] + 1

Variables
:   * **weight** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the learnable weights of the module of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</mrow>
<annotation encoding="application/x-tex">
               (text{in_channels}, frac{text{out_channels}}{text{groups}},
              </annotation>
</semantics>
</math> -->( in_channels , out_channels groups , (text{in_channels}, frac{text{out_channels}}{text{groups}},( in_channels , groups out_channels ​ , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
                kernel_size[0]
               </mtext>
<mo separator="true">
                ,
               </mo>
<mtext>
                kernel_size[1]
               </mtext>
<mo separator="true">
                ,
               </mo>
<mtext>
                kernel_size[2]
               </mtext>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               text{kernel_size[0]}, text{kernel_size[1]}, text{kernel_size[2]})
              </annotation>
</semantics>
</math> -->kernel_size[0] , kernel_size[1] , kernel_size[2] ) text{kernel_size[0]}, text{kernel_size[1]}, text{kernel_size[2]})kernel_size[0] , kernel_size[1] , kernel_size[2] )  .
The values of these weights are sampled from <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="script">
                U
               </mi>
<mo stretchy="false">
                (
               </mo>
<mo>
                −
               </mo>
<msqrt>
<mi>
                 k
                </mi>
</msqrt>
<mo separator="true">
                ,
               </mo>
<msqrt>
<mi>
                 k
                </mi>
</msqrt>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               mathcal{U}(-sqrt{k}, sqrt{k})
              </annotation>
</semantics>
</math> -->U ( − k , k ) mathcal{U}(-sqrt{k}, sqrt{k})U ( − k ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ , k ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                k
               </mi>
<mo>
                =
               </mo>
<mfrac>
<mrow>
<mi>
                  g
                 </mi>
<mi>
                  r
                 </mi>
<mi>
                  o
                 </mi>
<mi>
                  u
                 </mi>
<mi>
                  p
                 </mi>
<mi>
                  s
                 </mi>
</mrow>
<mrow>
<msub>
<mi>
                   C
                  </mi>
<mtext>
                   out
                  </mtext>
</msub>
<mo>
                  ∗
                 </mo>
<msubsup>
<mo>
                   ∏
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
<mn>
                   2
                  </mn>
</msubsup>
<mtext>
                  kernel_size
                 </mtext>
<mo stretchy="false">
                  [
                 </mo>
<mi>
                  i
                 </mi>
<mo stretchy="false">
                  ]
                 </mo>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
               k = frac{groups}{C_text{out} * prod_{i=0}^{2}text{kernel_size}[i]}
              </annotation>
</semantics>
</math> -->k = g r o u p s C out ∗ ∏ i = 0 2 kernel_size [ i ] k = frac{groups}{C_text{out} * prod_{i=0}^{2}text{kernel_size}[i]}k = C out ​ ∗ ∏ i = 0 2 ​ kernel_size [ i ] g ro u p s ​

* **bias** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the learnable bias of the module of shape (out_channels)
If `bias`  is `True`  , then the values of these weights are
sampled from <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="script">
                U
               </mi>
<mo stretchy="false">
                (
               </mo>
<mo>
                −
               </mo>
<msqrt>
<mi>
                 k
                </mi>
</msqrt>
<mo separator="true">
                ,
               </mo>
<msqrt>
<mi>
                 k
                </mi>
</msqrt>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               mathcal{U}(-sqrt{k}, sqrt{k})
              </annotation>
</semantics>
</math> -->U ( − k , k ) mathcal{U}(-sqrt{k}, sqrt{k})U ( − k ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ , k ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                k
               </mi>
<mo>
                =
               </mo>
<mfrac>
<mrow>
<mi>
                  g
                 </mi>
<mi>
                  r
                 </mi>
<mi>
                  o
                 </mi>
<mi>
                  u
                 </mi>
<mi>
                  p
                 </mi>
<mi>
                  s
                 </mi>
</mrow>
<mrow>
<msub>
<mi>
                   C
                  </mi>
<mtext>
                   out
                  </mtext>
</msub>
<mo>
                  ∗
                 </mo>
<msubsup>
<mo>
                   ∏
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
<mn>
                   2
                  </mn>
</msubsup>
<mtext>
                  kernel_size
                 </mtext>
<mo stretchy="false">
                  [
                 </mo>
<mi>
                  i
                 </mi>
<mo stretchy="false">
                  ]
                 </mo>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
               k = frac{groups}{C_text{out} * prod_{i=0}^{2}text{kernel_size}[i]}
              </annotation>
</semantics>
</math> -->k = g r o u p s C out ∗ ∏ i = 0 2 kernel_size [ i ] k = frac{groups}{C_text{out} * prod_{i=0}^{2}text{kernel_size}[i]}k = C out ​ ∗ ∏ i = 0 2 ​ kernel_size [ i ] g ro u p s ​

Examples: 

```
>>> # With square kernels and equal stride
>>> m = nn.ConvTranspose3d(16, 33, 3, stride=2)
>>> # non-square kernels and unequal stride and with padding
>>> m = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
>>> input = torch.randn(20, 16, 10, 50, 100)
>>> output = m(input)

```

