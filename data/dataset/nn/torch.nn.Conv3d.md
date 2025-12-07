Conv3d 
================================================

*class* torch.nn. Conv3d ( *in_channels*  , *out_channels*  , *kernel_size*  , *stride = 1*  , *padding = 0*  , *dilation = 1*  , *groups = 1*  , *bias = True*  , *padding_mode = 'zeros'*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/conv.py#L551) 
:   Applies a 3D convolution over an input signal composed of several input
planes. 

In the simplest case, the output value of the layer with input size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mi>
            D
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            H
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            W
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (N, C_{in}, D, H, W)
          </annotation>
</semantics>
</math> -->( N , C i n , D , H , W ) (N, C_{in}, D, H, W)( N , C in ​ , D , H , W )  and output <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N , C o u t , D o u t , H o u t , W o u t ) (N, C_{out}, D_{out}, H_{out}, W_{out})( N , C o u t ​ , D o u t ​ , H o u t ​ , W o u t ​ )  can be precisely described as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
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
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             N
            </mi>
<mi>
             i
            </mi>
</msub>
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
<msub>
<mi>
               t
              </mi>
<mi>
               j
              </mi>
</msub>
</mrow>
</msub>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mi>
            b
           </mi>
<mi>
            i
           </mi>
<mi>
            a
           </mi>
<mi>
            s
           </mi>
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
<msub>
<mi>
               t
              </mi>
<mi>
               j
              </mi>
</msub>
</mrow>
</msub>
<mo stretchy="false">
            )
           </mo>
<mo>
            +
           </mo>
<munderover>
<mo>
             ∑
            </mo>
<mrow>
<mi>
              k
             </mi>
<mo>
              =
             </mo>
<mn>
              0
             </mn>
</mrow>
<mrow>
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
<mo>
              −
             </mo>
<mn>
              1
             </mn>
</mrow>
</munderover>
<mi>
            w
           </mi>
<mi>
            e
           </mi>
<mi>
            i
           </mi>
<mi>
            g
           </mi>
<mi>
            h
           </mi>
<mi>
            t
           </mi>
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
<msub>
<mi>
               t
              </mi>
<mi>
               j
              </mi>
</msub>
</mrow>
</msub>
<mo separator="true">
            ,
           </mo>
<mi>
            k
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            ⋆
           </mo>
<mi>
            i
           </mi>
<mi>
            n
           </mi>
<mi>
            p
           </mi>
<mi>
            u
           </mi>
<mi>
            t
           </mi>
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             N
            </mi>
<mi>
             i
            </mi>
</msub>
<mo separator="true">
            ,
           </mo>
<mi>
            k
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           out(N_i, C_{out_j}) = bias(C_{out_j}) +
                        sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) star input(N_i, k)
          </annotation>
</semantics>
</math> -->
o u t ( N i , C o u t j ) = b i a s ( C o u t j ) + ∑ k = 0 C i n − 1 w e i g h t ( C o u t j , k ) ⋆ i n p u t ( N i , k ) out(N_i, C_{out_j}) = bias(C_{out_j}) +
 sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) star input(N_i, k)

o u t ( N i ​ , C o u t j ​ ​ ) = bia s ( C o u t j ​ ​ ) + k = 0 ∑ C in ​ − 1 ​ w e i g h t ( C o u t j ​ ​ , k ) ⋆ in p u t ( N i ​ , k )

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
            ⋆
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           star
          </annotation>
</semantics>
</math> -->⋆ star⋆  is the valid 3D [cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation)  operator 

This module supports [TensorFloat32](../notes/cuda.html#tf32-on-ampere)  . 

On certain ROCm devices, when using float16 inputs this module will use [different precision](../notes/numerical_accuracy.html#fp16-on-mi200)  for backward. 

* `stride`  controls the stride for the cross-correlation.
* `padding`  controls the amount of padding applied to the input. It
can be either a string {‘valid’, ‘same’} or a tuple of ints giving the
amount of implicit padding applied on both sides.
* `dilation`  controls the spacing between the kernel points; also known as the à trous algorithm.
It is harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)  has a nice visualization of what `dilation`  does.
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

The parameters `kernel_size`  , `stride`  , `padding`  , `dilation`  can either be: 

> * a single `int`  – in which case the same value is used for the depth, height and width dimension
> * a `tuple`  of three ints – in which case, the first *int* is used for the depth dimension,
> the second *int* for the height dimension and the third *int* for the width dimension

Note 

When *groups == in_channels* and *out_channels == K * in_channels* ,
where *K* is a positive integer, this operation is also known as a “depthwise convolution”. 

In other words, for an input of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
              L
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
            (N, C_{in}, L_{in})
           </annotation>
</semantics>
</math> -->( N , C i n , L i n ) (N, C_{in}, L_{in})( N , C in ​ , L in ​ )  ,
a depthwise convolution with a depthwise multiplier *K* can be performed with the arguments <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
             (
            </mo>
<msub>
<mi>
              C
             </mi>
<mtext>
              in
             </mtext>
</msub>
<mo>
             =
            </mo>
<msub>
<mi>
              C
             </mi>
<mtext>
              in
             </mtext>
</msub>
<mo separator="true">
             ,
            </mo>
<msub>
<mi>
              C
             </mi>
<mtext>
              out
             </mtext>
</msub>
<mo>
             =
            </mo>
<msub>
<mi>
              C
             </mi>
<mtext>
              in
             </mtext>
</msub>
<mo>
             ×
            </mo>
<mtext>
             K
            </mtext>
<mo separator="true">
             ,
            </mo>
<mi mathvariant="normal">
             .
            </mi>
<mi mathvariant="normal">
             .
            </mi>
<mi mathvariant="normal">
             .
            </mi>
<mo separator="true">
             ,
            </mo>
<mtext>
             groups
            </mtext>
<mo>
             =
            </mo>
<msub>
<mi>
              C
             </mi>
<mtext>
              in
             </mtext>
</msub>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            (C_text{in}=C_text{in}, C_text{out}=C_text{in} times text{K}, ..., text{groups}=C_text{in})
           </annotation>
</semantics>
</math> -->( C in = C in , C out = C in × K , . . . , groups = C in ) (C_text{in}=C_text{in}, C_text{out}=C_text{in} times text{K}, ..., text{groups}=C_text{in})( C in ​ = C in ​ , C out ​ = C in ​ × K , ... , groups = C in ​ )  .

Note 

In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting `torch.backends.cudnn.deterministic = True`  . See [Reproducibility](../notes/randomness.html)  for more information.

Note 

`padding='valid'`  is the same as no padding. `padding='same'`  pads
the input so the output has the shape as the input. However, this mode
doesn’t support any stride values other than 1.

Note 

This module supports complex data types i.e. `complex32, complex64, complex128`  .

Parameters
:   * **in_channels** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Number of channels in the input image
* **out_channels** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Number of channels produced by the convolution
* **kernel_size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  ) – Size of the convolving kernel
* **stride** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – Stride of the convolution. Default: 1
* **padding** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – Padding added to all six sides of
the input. Default: 0
* **dilation** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – Spacing between kernel elements. Default: 1
* **groups** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – Number of blocked connections from input channels to output channels. Default: 1
* **bias** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If `True`  , adds a learnable bias to the output. Default: `True`
* **padding_mode** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – `'zeros'`  , `'reflect'`  , `'replicate'`  or `'circular'`  . Default: `'zeros'`

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
</math> -->( C o u t , D o u t , H o u t , W o u t ) (C_{out}, D_{out}, H_{out}, W_{out})( C o u t ​ , D o u t ​ , H o u t ​ , W o u t ​ )  ,
where

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
    <mrow>
    <mo fence="true">
                     ⌊
                    </mo>
    <mfrac>
    <mrow>
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
                       +
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
                       −
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
                       −
                      </mo>
    <mn>
                       1
                      </mn>
    </mrow>
    <mrow>
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
    </mrow>
    </mfrac>
    <mo>
                     +
                    </mo>
    <mn>
                     1
                    </mn>
    <mo fence="true">
                     ⌋
                    </mo>
    </mrow>
    </mrow>
    <annotation encoding="application/x-tex">
                   D_{out} = leftlfloorfrac{D_{in} + 2 times text{padding}[0] - text{dilation}[0]
          times (text{kernel_size}[0] - 1) - 1}{text{stride}[0]} + 1rightrfloor
                  </annotation>
    </semantics>
    </math> -->
    D o u t = ⌊ D i n + 2 × padding [ 0 ] − dilation [ 0 ] × ( kernel_size [ 0 ] − 1 ) − 1 stride [ 0 ] + 1 ⌋ D_{out} = leftlfloorfrac{D_{in} + 2 times text{padding}[0] - text{dilation}[0]
     times (text{kernel_size}[0] - 1) - 1}{text{stride}[0]} + 1rightrfloor

    D o u t ​ = ⌊ stride [ 0 ] D in ​ + 2 × padding [ 0 ] − dilation [ 0 ] × ( kernel_size [ 0 ] − 1 ) − 1 ​ + 1 ⌋

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
        <mrow>
        <mo fence="true">
                         ⌊
                        </mo>
        <mfrac>
        <mrow>
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
                           +
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
                           −
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
                           −
                          </mo>
        <mn>
                           1
                          </mn>
        </mrow>
        <mrow>
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
        </mrow>
        </mfrac>
        <mo>
                         +
                        </mo>
        <mn>
                         1
                        </mn>
        <mo fence="true">
                         ⌋
                        </mo>
        </mrow>
        </mrow>
        <annotation encoding="application/x-tex">
                       H_{out} = leftlfloorfrac{H_{in} + 2 times text{padding}[1] - text{dilation}[1]
              times (text{kernel_size}[1] - 1) - 1}{text{stride}[1]} + 1rightrfloor
                      </annotation>
        </semantics>
        </math> -->
        H o u t = ⌊ H i n + 2 × padding [ 1 ] − dilation [ 1 ] × ( kernel_size [ 1 ] − 1 ) − 1 stride [ 1 ] + 1 ⌋ H_{out} = leftlfloorfrac{H_{in} + 2 times text{padding}[1] - text{dilation}[1]
         times (text{kernel_size}[1] - 1) - 1}{text{stride}[1]} + 1rightrfloor

    H o u t ​ = ⌊ stride [ 1 ] H in ​ + 2 × padding [ 1 ] − dilation [ 1 ] × ( kernel_size [ 1 ] − 1 ) − 1 ​ + 1 ⌋

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
        <mrow>
        <mo fence="true">
                         ⌊
                        </mo>
        <mfrac>
        <mrow>
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
                           +
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
                           −
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
                           −
                          </mo>
        <mn>
                           1
                          </mn>
        </mrow>
        <mrow>
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
        </mrow>
        </mfrac>
        <mo>
                         +
                        </mo>
        <mn>
                         1
                        </mn>
        <mo fence="true">
                         ⌋
                        </mo>
        </mrow>
        </mrow>
        <annotation encoding="application/x-tex">
                       W_{out} = leftlfloorfrac{W_{in} + 2 times text{padding}[2] - text{dilation}[2]
              times (text{kernel_size}[2] - 1) - 1}{text{stride}[2]} + 1rightrfloor
                      </annotation>
        </semantics>
        </math> -->
        W o u t = ⌊ W i n + 2 × padding [ 2 ] − dilation [ 2 ] × ( kernel_size [ 2 ] − 1 ) − 1 stride [ 2 ] + 1 ⌋ W_{out} = leftlfloorfrac{W_{in} + 2 times text{padding}[2] - text{dilation}[2]
         times (text{kernel_size}[2] - 1) - 1}{text{stride}[2]} + 1rightrfloor

    W o u t ​ = ⌊ stride [ 2 ] W in ​ + 2 × padding [ 2 ] − dilation [ 2 ] × ( kernel_size [ 2 ] − 1 ) − 1 ​ + 1 ⌋

Variables
:   * **weight** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the learnable weights of the module of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mtext>
                out_channels
               </mtext>
<mo separator="true">
                ,
               </mo>
<mfrac>
<mtext>
                 in_channels
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
               (text{out_channels}, frac{text{in_channels}}{text{groups}},
              </annotation>
</semantics>
</math> -->( out_channels , in_channels groups , (text{out_channels}, frac{text{in_channels}}{text{groups}},( out_channels , groups in_channels ​ , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                   in
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
               k = frac{groups}{C_text{in} * prod_{i=0}^{2}text{kernel_size}[i]}
              </annotation>
</semantics>
</math> -->k = g r o u p s C in ∗ ∏ i = 0 2 kernel_size [ i ] k = frac{groups}{C_text{in} * prod_{i=0}^{2}text{kernel_size}[i]}k = C in ​ ∗ ∏ i = 0 2 ​ kernel_size [ i ] g ro u p s ​

* **bias** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the learnable bias of the module of shape (out_channels). If `bias`  is `True`  ,
then the values of these weights are
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
                   in
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
               k = frac{groups}{C_text{in} * prod_{i=0}^{2}text{kernel_size}[i]}
              </annotation>
</semantics>
</math> -->k = g r o u p s C in ∗ ∏ i = 0 2 kernel_size [ i ] k = frac{groups}{C_text{in} * prod_{i=0}^{2}text{kernel_size}[i]}k = C in ​ ∗ ∏ i = 0 2 ​ kernel_size [ i ] g ro u p s ​

Examples: 

```
>>> # With square kernels and equal stride
>>> m = nn.Conv3d(16, 33, 3, stride=2)
>>> # non-square kernels and unequal stride and with padding
>>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
>>> input = torch.randn(20, 16, 10, 50, 100)
>>> output = m(input)

```

