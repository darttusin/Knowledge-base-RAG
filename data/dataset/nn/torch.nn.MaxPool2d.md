MaxPool2d 
======================================================

*class* torch.nn. MaxPool2d ( *kernel_size*  , *stride = None*  , *padding = 0*  , *dilation = 1*  , *return_indices = False*  , *ceil_mode = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/pooling.py#L156) 
:   Applies a 2D max pooling over an input signal composed of several input planes. 

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
<mi>
            C
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
           (N, C, H, W)
          </annotation>
</semantics>
</math> -->( N , C , H , W ) (N, C, H, W)( N , C , H , W )  ,
output <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mi>
            C
           </mi>
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
           (N, C, H_{out}, W_{out})
          </annotation>
</semantics>
</math> -->( N , C , H o u t , W o u t ) (N, C, H_{out}, W_{out})( N , C , H o u t ​ , W o u t ​ )  and `kernel_size` <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
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
           (kH, kW)
          </annotation>
</semantics>
</math> -->( k H , k W ) (kH, kW)( k H , kW )  can be precisely described as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right left" columnspacing="0em" rowspacing="0.25em">
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
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
<mi>
                 j
                </mi>
</msub>
<mo separator="true">
                ,
               </mo>
<mi>
                h
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                w
               </mi>
<mo stretchy="false">
                )
               </mo>
<mo>
                =
               </mo>
<mrow>
</mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<munder>
<mrow>
<mi>
                  max
                 </mi>
<mo>
                  ⁡
                 </mo>
</mrow>
<mrow>
<mi>
                  m
                 </mi>
<mo>
                  =
                 </mo>
<mn>
                  0
                 </mn>
<mo separator="true">
                  ,
                 </mo>
<mo>
                  …
                 </mo>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  k
                 </mi>
<mi>
                  H
                 </mi>
<mo>
                  −
                 </mo>
<mn>
                  1
                 </mn>
</mrow>
</munder>
<munder>
<mrow>
<mi>
                  max
                 </mi>
<mo>
                  ⁡
                 </mo>
</mrow>
<mrow>
<mi>
                  n
                 </mi>
<mo>
                  =
                 </mo>
<mn>
                  0
                 </mn>
<mo separator="true">
                  ,
                 </mo>
<mo>
                  …
                 </mo>
<mo separator="true">
                  ,
                 </mo>
<mi>
                  k
                 </mi>
<mi>
                  W
                 </mi>
<mo>
                  −
                 </mo>
<mn>
                  1
                 </mn>
</mrow>
</munder>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mtext>
                input
               </mtext>
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
<mi>
                 j
                </mi>
</msub>
<mo separator="true">
                ,
               </mo>
<mtext>
                stride[0]
               </mtext>
<mo>
                ×
               </mo>
<mi>
                h
               </mi>
<mo>
                +
               </mo>
<mi>
                m
               </mi>
<mo separator="true">
                ,
               </mo>
<mtext>
                stride[1]
               </mtext>
<mo>
                ×
               </mo>
<mi>
                w
               </mi>
<mo>
                +
               </mo>
<mi>
                n
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
           begin{aligned}
    out(N_i, C_j, h, w) ={} &amp; max_{m=0, ldots, kH-1} max_{n=0, ldots, kW-1} 
                            &amp; text{input}(N_i, C_j, text{stride[0]} times h + m,
                                           text{stride[1]} times w + n)
end{aligned}
          </annotation>
</semantics>
</math> -->
o u t ( N i , C j , h , w ) = max ⁡ m = 0 , … , k H − 1 max ⁡ n = 0 , … , k W − 1 input ( N i , C j , stride[0] × h + m , stride[1] × w + n ) begin{aligned}
 out(N_i, C_j, h, w) ={} & max_{m=0, ldots, kH-1} max_{n=0, ldots, kW-1} 
 & text{input}(N_i, C_j, text{stride[0]} times h + m,
 text{stride[1]} times w + n)
end{aligned}

o u t ( N i ​ , C j ​ , h , w ) = ​ m = 0 , … , k H − 1 max ​ n = 0 , … , kW − 1 max ​ input ( N i ​ , C j ​ , stride[0] × h + m , stride[1] × w + n ) ​

If `padding`  is non-zero, then the input is implicitly padded with negative infinity on both sides
for `padding`  number of points. `dilation`  controls the spacing between the kernel points.
It is harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)  has a nice visualization of what `dilation`  does. 

Note 

When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
or the input. Sliding windows that would start in the right padded region are ignored.

The parameters `kernel_size`  , `stride`  , `padding`  , `dilation`  can either be: 

> * a single `int`  – in which case the same value is used for the height and width dimension
> * a `tuple`  of two ints – in which case, the first *int* is used for the height dimension,
> and the second *int* for the width dimension

Parameters
:   * **kernel_size** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – the size of the window to take a max over
* **stride** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – the stride of the window. Default value is `kernel_size`
* **padding** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – Implicit negative infinity padding to be added on both sides
* **dilation** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – a parameter that controls the stride of elements in the window
* **return_indices** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – if `True`  , will return the max indices along with the outputs.
Useful for [`torch.nn.MaxUnpool2d`](torch.nn.MaxUnpool2d.html#torch.nn.MaxUnpool2d "torch.nn.MaxUnpool2d")  later
* **ceil_mode** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – when True, will use *ceil* instead of *floor* to compute the output shape

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
<mi>
                C
               </mi>
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
               (N, C, H_{in}, W_{in})
              </annotation>
</semantics>
</math> -->( N , C , H i n , W i n ) (N, C, H_{in}, W_{in})( N , C , H in ​ , W in ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                C
               </mi>
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
               (C, H_{in}, W_{in})
              </annotation>
</semantics>
</math> -->( C , H i n , W i n ) (C, H_{in}, W_{in})( C , H in ​ , W in ​ )

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
<mi>
                C
               </mi>
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
               (N, C, H_{out}, W_{out})
              </annotation>
</semantics>
</math> -->( N , C , H o u t , W o u t ) (N, C, H_{out}, W_{out})( N , C , H o u t ​ , W o u t ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                C
               </mi>
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
               (C, H_{out}, W_{out})
              </annotation>
</semantics>
</math> -->( C , H o u t , W o u t ) (C, H_{out}, W_{out})( C , H o u t ​ , W o u t ​ )  , where

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
                       ∗
                      </mo>
    <mtext>
                       padding[0]
                      </mtext>
    <mo>
                       −
                      </mo>
    <mtext>
                       dilation[0]
                      </mtext>
    <mo>
                       ×
                      </mo>
    <mo stretchy="false">
                       (
                      </mo>
    <mtext>
                       kernel_size[0]
                      </mtext>
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
    <mtext>
                      stride[0]
                     </mtext>
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
                   H_{out} = leftlfloorfrac{H_{in} + 2 * text{padding[0]} - text{dilation[0]}
          times (text{kernel_size[0]} - 1) - 1}{text{stride[0]}} + 1rightrfloor
                  </annotation>
    </semantics>
    </math> -->
    H o u t = ⌊ H i n + 2 ∗ padding[0] − dilation[0] × ( kernel_size[0] − 1 ) − 1 stride[0] + 1 ⌋ H_{out} = leftlfloorfrac{H_{in} + 2 * text{padding[0]} - text{dilation[0]}
     times (text{kernel_size[0]} - 1) - 1}{text{stride[0]}} + 1rightrfloor

    H o u t ​ = ⌊ stride[0] H in ​ + 2 ∗ padding[0] − dilation[0] × ( kernel_size[0] − 1 ) − 1 ​ + 1 ⌋

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
                           ∗
                          </mo>
        <mtext>
                           padding[1]
                          </mtext>
        <mo>
                           −
                          </mo>
        <mtext>
                           dilation[1]
                          </mtext>
        <mo>
                           ×
                          </mo>
        <mo stretchy="false">
                           (
                          </mo>
        <mtext>
                           kernel_size[1]
                          </mtext>
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
        <mtext>
                          stride[1]
                         </mtext>
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
                       W_{out} = leftlfloorfrac{W_{in} + 2 * text{padding[1]} - text{dilation[1]}
              times (text{kernel_size[1]} - 1) - 1}{text{stride[1]}} + 1rightrfloor
                      </annotation>
        </semantics>
        </math> -->
        W o u t = ⌊ W i n + 2 ∗ padding[1] − dilation[1] × ( kernel_size[1] − 1 ) − 1 stride[1] + 1 ⌋ W_{out} = leftlfloorfrac{W_{in} + 2 * text{padding[1]} - text{dilation[1]}
         times (text{kernel_size[1]} - 1) - 1}{text{stride[1]}} + 1rightrfloor

    W o u t ​ = ⌊ stride[1] W in ​ + 2 ∗ padding[1] − dilation[1] × ( kernel_size[1] − 1 ) − 1 ​ + 1 ⌋

Examples: 

```
>>> # pool of square window of size=3, stride=2
>>> m = nn.MaxPool2d(3, stride=2)
>>> # pool of non-square window
>>> m = nn.MaxPool2d((3, 2), stride=(2, 1))
>>> input = torch.randn(20, 16, 50, 32)
>>> output = m(input)

```

