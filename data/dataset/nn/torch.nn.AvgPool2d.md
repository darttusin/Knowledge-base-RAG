AvgPool2d 
======================================================

*class* torch.nn. AvgPool2d ( *kernel_size*  , *stride = None*  , *padding = 0*  , *ceil_mode = False*  , *count_include_pad = True*  , *divisor_override = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/pooling.py#L675) 
:   Applies a 2D average pooling over an input signal composed of several input planes. 

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
<mfrac>
<mn>
             1
            </mn>
<mrow>
<mi>
              k
             </mi>
<mi>
              H
             </mi>
<mo>
              ∗
             </mo>
<mi>
              k
             </mi>
<mi>
              W
             </mi>
</mrow>
</mfrac>
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
            s
           </mi>
<mi>
            t
           </mi>
<mi>
            r
           </mi>
<mi>
            i
           </mi>
<mi>
            d
           </mi>
<mi>
            e
           </mi>
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
<mi>
            s
           </mi>
<mi>
            t
           </mi>
<mi>
            r
           </mi>
<mi>
            i
           </mi>
<mi>
            d
           </mi>
<mi>
            e
           </mi>
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
<annotation encoding="application/x-tex">
           out(N_i, C_j, h, w)  = frac{1}{kH * kW} sum_{m=0}^{kH-1} sum_{n=0}^{kW-1}
                       input(N_i, C_j, stride[0] times h + m, stride[1] times w + n)
          </annotation>
</semantics>
</math> -->
o u t ( N i , C j , h , w ) = 1 k H ∗ k W ∑ m = 0 k H − 1 ∑ n = 0 k W − 1 i n p u t ( N i , C j , s t r i d e [ 0 ] × h + m , s t r i d e [ 1 ] × w + n ) out(N_i, C_j, h, w) = frac{1}{kH * kW} sum_{m=0}^{kH-1} sum_{n=0}^{kW-1}
 input(N_i, C_j, stride[0] times h + m, stride[1] times w + n)

o u t ( N i ​ , C j ​ , h , w ) = k H ∗ kW 1 ​ m = 0 ∑ k H − 1 ​ n = 0 ∑ kW − 1 ​ in p u t ( N i ​ , C j ​ , s t r i d e [ 0 ] × h + m , s t r i d e [ 1 ] × w + n )

If `padding`  is non-zero, then the input is implicitly zero-padded on both sides
for `padding`  number of points. 

Note 

When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
or the input. Sliding windows that would start in the right padded region are ignored.

Note 

pad should be at most half of effective kernel size.

The parameters `kernel_size`  , `stride`  , `padding`  can either be: 

> * a single `int`  or a single-element tuple – in which case the same value is used for the height and width dimension
> * a `tuple`  of two ints – in which case, the first *int* is used for the height dimension,
> and the second *int* for the width dimension

Parameters
:   * **kernel_size** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – the size of the window
* **stride** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – the stride of the window. Default value is `kernel_size`
* **padding** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – implicit zero padding to be added on both sides
* **ceil_mode** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – when True, will use *ceil* instead of *floor* to compute the output shape
* **count_include_pad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – when True, will include the zero-padding in the averaging calculation
* **divisor_override** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]*  ) – if specified, it will be used as divisor, otherwise size of the pooling region will be used.

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
</math> -->( C , H i n , W i n ) (C, H_{in}, W_{in})( C , H in ​ , W in ​ )  .

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
                   H_{out} = leftlfloorfrac{H_{in}  + 2 times text{padding}[0] -
      text{kernel_size}[0]}{text{stride}[0]} + 1rightrfloor
                  </annotation>
    </semantics>
    </math> -->
    H o u t = ⌊ H i n + 2 × padding [ 0 ] − kernel_size [ 0 ] stride [ 0 ] + 1 ⌋ H_{out} = leftlfloorfrac{H_{in} + 2 times text{padding}[0] -
     text{kernel_size}[0]}{text{stride}[0]} + 1rightrfloor

    H o u t ​ = ⌊ stride [ 0 ] H in ​ + 2 × padding [ 0 ] − kernel_size [ 0 ] ​ + 1 ⌋

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
                       W_{out} = leftlfloorfrac{W_{in}  + 2 times text{padding}[1] -
          text{kernel_size}[1]}{text{stride}[1]} + 1rightrfloor
                      </annotation>
        </semantics>
        </math> -->
        W o u t = ⌊ W i n + 2 × padding [ 1 ] − kernel_size [ 1 ] stride [ 1 ] + 1 ⌋ W_{out} = leftlfloorfrac{W_{in} + 2 times text{padding}[1] -
         text{kernel_size}[1]}{text{stride}[1]} + 1rightrfloor

    W o u t ​ = ⌊ stride [ 1 ] W in ​ + 2 × padding [ 1 ] − kernel_size [ 1 ] ​ + 1 ⌋

    Per the note above, if `ceil_mode`  is True and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
        <semantics>
        <mrow>
        <mo stretchy="false">
                        (
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
                       (H_{out} - 1)times text{stride}[0]geq H_{in}

        + text{padding}[0]
                      </annotation>
        </semantics>
        </math> -->( H o u t − 1 ) × stride [ 0 ] ≥ H i n + padding [ 0 ] (H_{out} - 1)times text{stride}[0]geq H_{in}
        + text{padding}[0]( H o u t ​ − 1 ) × stride [ 0 ] ≥ H in ​ + padding [ 0 ]  , we skip the last window as it would start in the bottom padded region,
        resulting in <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
        </math> -->H o u t H_{out}H o u t ​  being reduced by one.

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
        </math> -->W o u t W_{out}W o u t ​  .

Examples: 

```
>>> # pool of square window of size=3, stride=2
>>> m = nn.AvgPool2d(3, stride=2)
>>> # pool of non-square window
>>> m = nn.AvgPool2d((3, 2), stride=(2, 1))
>>> input = torch.randn(20, 16, 50, 32)
>>> output = m(input)

```

