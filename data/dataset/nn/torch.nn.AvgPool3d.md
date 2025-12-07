AvgPool3d 
======================================================

*class* torch.nn. AvgPool3d ( *kernel_size*  , *stride = None*  , *padding = 0*  , *ceil_mode = False*  , *count_include_pad = True*  , *divisor_override = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/pooling.py#L784) 
:   Applies a 3D average pooling over an input signal composed of several input planes. 

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
</munderover>
<munderover>
<mo>
                 ∑
                </mo>
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
</mrow>
<mrow>
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
</munderover>
<munderover>
<mo>
                 ∑
                </mo>
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
</mrow>
<mrow>
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
</munderover>
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
<mfrac>
<mrow>
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
<mrow>
<mi>
                  k
                 </mi>
<mi>
                  D
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
</mfrac>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
           begin{aligned}
    text{out}(N_i, C_j, d, h, w) ={} &amp; sum_{k=0}^{kD-1} sum_{m=0}^{kH-1} sum_{n=0}^{kW-1} 
                                      &amp; frac{text{input}(N_i, C_j, text{stride}[0] times d + k,
                                              text{stride}[1] times h + m, text{stride}[2] times w + n)}
                                             {kD times kH times kW}
end{aligned}
          </annotation>
</semantics>
</math> -->
out ( N i , C j , d , h , w ) = ∑ k = 0 k D − 1 ∑ m = 0 k H − 1 ∑ n = 0 k W − 1 input ( N i , C j , stride [ 0 ] × d + k , stride [ 1 ] × h + m , stride [ 2 ] × w + n ) k D × k H × k W begin{aligned}
 text{out}(N_i, C_j, d, h, w) ={} & sum_{k=0}^{kD-1} sum_{m=0}^{kH-1} sum_{n=0}^{kW-1} 
 & frac{text{input}(N_i, C_j, text{stride}[0] times d + k,
 text{stride}[1] times h + m, text{stride}[2] times w + n)}
 {kD times kH times kW}
end{aligned}

out ( N i ​ , C j ​ , d , h , w ) = ​ k = 0 ∑ k D − 1 ​ m = 0 ∑ k H − 1 ​ n = 0 ∑ kW − 1 ​ k D × k H × kW input ( N i ​ , C j ​ , stride [ 0 ] × d + k , stride [ 1 ] × h + m , stride [ 2 ] × w + n ) ​ ​

If `padding`  is non-zero, then the input is implicitly zero-padded on all three sides
for `padding`  number of points. 

Note 

When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
or the input. Sliding windows that would start in the right padded region are ignored.

Note 

pad should be at most half of effective kernel size.

The parameters `kernel_size`  , `stride`  can either be: 

> * a single `int`  – in which case the same value is used for the depth, height and width dimension
> * a `tuple`  of three ints – in which case, the first *int* is used for the depth dimension,
> the second *int* for the height dimension and the third *int* for the width dimension

Parameters
:   * **kernel_size** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – the size of the window
* **stride** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – the stride of the window. Default value is `kernel_size`
* **padding** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – implicit zero padding to be added on all three sides
* **ceil_mode** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – when True, will use *ceil* instead of *floor* to compute the output shape
* **count_include_pad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – when True, will include the zero-padding in the averaging calculation
* **divisor_override** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]*  ) – if specified, it will be used as divisor, otherwise `kernel_size`  will be used

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
                   D_{out} = leftlfloorfrac{D_{in} + 2 times text{padding}[0] -
          text{kernel_size}[0]}{text{stride}[0]} + 1rightrfloor
                  </annotation>
    </semantics>
    </math> -->
    D o u t = ⌊ D i n + 2 × padding [ 0 ] − kernel_size [ 0 ] stride [ 0 ] + 1 ⌋ D_{out} = leftlfloorfrac{D_{in} + 2 times text{padding}[0] -
     text{kernel_size}[0]}{text{stride}[0]} + 1rightrfloor

    D o u t ​ = ⌊ stride [ 0 ] D in ​ + 2 × padding [ 0 ] − kernel_size [ 0 ] ​ + 1 ⌋

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
                       H_{out} = leftlfloorfrac{H_{in} + 2 times text{padding}[1] -
              text{kernel_size}[1]}{text{stride}[1]} + 1rightrfloor
                      </annotation>
        </semantics>
        </math> -->
        H o u t = ⌊ H i n + 2 × padding [ 1 ] − kernel_size [ 1 ] stride [ 1 ] + 1 ⌋ H_{out} = leftlfloorfrac{H_{in} + 2 times text{padding}[1] -
         text{kernel_size}[1]}{text{stride}[1]} + 1rightrfloor

    H o u t ​ = ⌊ stride [ 1 ] H in ​ + 2 × padding [ 1 ] − kernel_size [ 1 ] ​ + 1 ⌋

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
                       W_{out} = leftlfloorfrac{W_{in} + 2 times text{padding}[2] -
              text{kernel_size}[2]}{text{stride}[2]} + 1rightrfloor
                      </annotation>
        </semantics>
        </math> -->
        W o u t = ⌊ W i n + 2 × padding [ 2 ] − kernel_size [ 2 ] stride [ 2 ] + 1 ⌋ W_{out} = leftlfloorfrac{W_{in} + 2 times text{padding}[2] -
         text{kernel_size}[2]}{text{stride}[2]} + 1rightrfloor

    W o u t ​ = ⌊ stride [ 2 ] W in ​ + 2 × padding [ 2 ] − kernel_size [ 2 ] ​ + 1 ⌋

    Per the note above, if `ceil_mode`  is True and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
        <semantics>
        <mrow>
        <mo stretchy="false">
                        (
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
                        ≥
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
                        +
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
        </mrow>
        <annotation encoding="application/x-tex">
                       (D_{out} - 1)times text{stride}[0]geq D_{in}

        + text{padding}[0]
                      </annotation>
        </semantics>
        </math> -->( D o u t − 1 ) × stride [ 0 ] ≥ D i n + padding [ 0 ] (D_{out} - 1)times text{stride}[0]geq D_{in}
        + text{padding}[0]( D o u t ​ − 1 ) × stride [ 0 ] ≥ D in ​ + padding [ 0 ]  , we skip the last window as it would start in the padded region,
        resulting in <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
        </mrow>
        <annotation encoding="application/x-tex">
                       D_{out}
                      </annotation>
        </semantics>
        </math> -->D o u t D_{out}D o u t ​  being reduced by one.

    The same applies for <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
        </mrow>
        <annotation encoding="application/x-tex">
                       W_{out}
                      </annotation>
        </semantics>
        </math> -->W o u t W_{out}W o u t ​  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
        </mrow>
        <annotation encoding="application/x-tex">
                       H_{out}
                      </annotation>
        </semantics>
        </math> -->H o u t H_{out}H o u t ​  .

Examples: 

```
>>> # pool of square window of size=3, stride=2
>>> m = nn.AvgPool3d(3, stride=2)
>>> # pool of non-square window
>>> m = nn.AvgPool3d((3, 2, 2), stride=(2, 1, 2))
>>> input = torch.randn(20, 16, 50, 44, 31)
>>> output = m(input)

```

