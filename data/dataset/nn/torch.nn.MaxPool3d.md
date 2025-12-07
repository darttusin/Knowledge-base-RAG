MaxPool3d 
======================================================

*class* torch.nn. MaxPool3d ( *kernel_size*  , *stride = None*  , *padding = 0*  , *dilation = 1*  , *return_indices = False*  , *ceil_mode = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/pooling.py#L235) 
:   Applies a 3D max pooling over an input signal composed of several input planes. 

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
           (N, C, D, H, W)
          </annotation>
</semantics>
</math> -->( N , C , D , H , W ) (N, C, D, H, W)( N , C , D , H , W )  ,
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
           (N, C, D_{out}, H_{out}, W_{out})
          </annotation>
</semantics>
</math> -->( N , C , D o u t , H o u t , W o u t ) (N, C, D_{out}, H_{out}, W_{out})( N , C , D o u t ​ , H o u t ​ , W o u t ​ )  and `kernel_size` <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mi>
            k
           </mi>
<mi>
            D
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
           (kD, kH, kW)
          </annotation>
</semantics>
</math> -->( k D , k H , k W ) (kD, kH, kW)( k D , k H , kW )  can be precisely described as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right left" columnspacing="0em" rowspacing="0.25em">
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mtext>
                out
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
<mi>
                d
               </mi>
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
                  k
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
                  D
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
                d
               </mi>
<mo>
                +
               </mo>
<mi>
                k
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
                stride[2]
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
    text{out}(N_i, C_j, d, h, w) ={} &amp; max_{k=0, ldots, kD-1} max_{m=0, ldots, kH-1} max_{n=0, ldots, kW-1} 
                                      &amp; text{input}(N_i, C_j, text{stride[0]} times d + k,
                                                     text{stride[1]} times h + m, text{stride[2]} times w + n)
end{aligned}
          </annotation>
</semantics>
</math> -->
out ( N i , C j , d , h , w ) = max ⁡ k = 0 , … , k D − 1 max ⁡ m = 0 , … , k H − 1 max ⁡ n = 0 , … , k W − 1 input ( N i , C j , stride[0] × d + k , stride[1] × h + m , stride[2] × w + n ) begin{aligned}
 text{out}(N_i, C_j, d, h, w) ={} & max_{k=0, ldots, kD-1} max_{m=0, ldots, kH-1} max_{n=0, ldots, kW-1} 
 & text{input}(N_i, C_j, text{stride[0]} times d + k,
 text{stride[1]} times h + m, text{stride[2]} times w + n)
end{aligned}

out ( N i ​ , C j ​ , d , h , w ) = ​ k = 0 , … , k D − 1 max ​ m = 0 , … , k H − 1 max ​ n = 0 , … , kW − 1 max ​ input ( N i ​ , C j ​ , stride[0] × d + k , stride[1] × h + m , stride[2] × w + n ) ​

If `padding`  is non-zero, then the input is implicitly padded with negative infinity on both sides
for `padding`  number of points. `dilation`  controls the spacing between the kernel points.
It is harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)  has a nice visualization of what `dilation`  does. 

Note 

When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
or the input. Sliding windows that would start in the right padded region are ignored.

The parameters `kernel_size`  , `stride`  , `padding`  , `dilation`  can either be: 

> * a single `int`  – in which case the same value is used for the depth, height and width dimension
> * a `tuple`  of three ints – in which case, the first *int* is used for the depth dimension,
> the second *int* for the height dimension and the third *int* for the width dimension

Parameters
:   * **kernel_size** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – the size of the window to take a max over
* **stride** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – the stride of the window. Default value is `kernel_size`
* **padding** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – Implicit negative infinity padding to be added on all three sides
* **dilation** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – a parameter that controls the stride of elements in the window
* **return_indices** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – if `True`  , will return the max indices along with the outputs.
Useful for [`torch.nn.MaxUnpool3d`](torch.nn.MaxUnpool3d.html#torch.nn.MaxUnpool3d "torch.nn.MaxUnpool3d")  later
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
               (N, C, D_{in}, H_{in}, W_{in})
              </annotation>
</semantics>
</math> -->( N , C , D i n , H i n , W i n ) (N, C, D_{in}, H_{in}, W_{in})( N , C , D in ​ , H in ​ , W in ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
               (C, D_{in}, H_{in}, W_{in})
              </annotation>
</semantics>
</math> -->( C , D i n , H i n , W i n ) (C, D_{in}, H_{in}, W_{in})( C , D in ​ , H in ​ , W in ​ )  .

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
               (N, C, D_{out}, H_{out}, W_{out})
              </annotation>
</semantics>
</math> -->( N , C , D o u t , H o u t , W o u t ) (N, C, D_{out}, H_{out}, W_{out})( N , C , D o u t ​ , H o u t ​ , W o u t ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
               (C, D_{out}, H_{out}, W_{out})
              </annotation>
</semantics>
</math> -->( C , D o u t , H o u t , W o u t ) (C, D_{out}, H_{out}, W_{out})( C , D o u t ​ , H o u t ​ , W o u t ​ )  , where

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
                   D_{out} = leftlfloorfrac{D_{in} + 2 times text{padding}[0] - text{dilation}[0] times
      (text{kernel_size}[0] - 1) - 1}{text{stride}[0]} + 1rightrfloor
                  </annotation>
    </semantics>
    </math> -->
    D o u t = ⌊ D i n + 2 × padding [ 0 ] − dilation [ 0 ] × ( kernel_size [ 0 ] − 1 ) − 1 stride [ 0 ] + 1 ⌋ D_{out} = leftlfloorfrac{D_{in} + 2 times text{padding}[0] - text{dilation}[0] times
     (text{kernel_size}[0] - 1) - 1}{text{stride}[0]} + 1rightrfloor

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
                       H_{out} = leftlfloorfrac{H_{in} + 2 times text{padding}[1] - text{dilation}[1] times
          (text{kernel_size}[1] - 1) - 1}{text{stride}[1]} + 1rightrfloor
                      </annotation>
        </semantics>
        </math> -->
        H o u t = ⌊ H i n + 2 × padding [ 1 ] − dilation [ 1 ] × ( kernel_size [ 1 ] − 1 ) − 1 stride [ 1 ] + 1 ⌋ H_{out} = leftlfloorfrac{H_{in} + 2 times text{padding}[1] - text{dilation}[1] times
         (text{kernel_size}[1] - 1) - 1}{text{stride}[1]} + 1rightrfloor

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
                       W_{out} = leftlfloorfrac{W_{in} + 2 times text{padding}[2] - text{dilation}[2] times
          (text{kernel_size}[2] - 1) - 1}{text{stride}[2]} + 1rightrfloor
                      </annotation>
        </semantics>
        </math> -->
        W o u t = ⌊ W i n + 2 × padding [ 2 ] − dilation [ 2 ] × ( kernel_size [ 2 ] − 1 ) − 1 stride [ 2 ] + 1 ⌋ W_{out} = leftlfloorfrac{W_{in} + 2 times text{padding}[2] - text{dilation}[2] times
         (text{kernel_size}[2] - 1) - 1}{text{stride}[2]} + 1rightrfloor

    W o u t ​ = ⌊ stride [ 2 ] W in ​ + 2 × padding [ 2 ] − dilation [ 2 ] × ( kernel_size [ 2 ] − 1 ) − 1 ​ + 1 ⌋

Examples: 

```
>>> # pool of square window of size=3, stride=2
>>> m = nn.MaxPool3d(3, stride=2)
>>> # pool of non-square window
>>> m = nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2))
>>> input = torch.randn(20, 16, 50, 44, 31)
>>> output = m(input)

```

