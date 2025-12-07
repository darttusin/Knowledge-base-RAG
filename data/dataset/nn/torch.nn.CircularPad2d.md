CircularPad2d 
==============================================================

*class* torch.nn. CircularPad2d ( *padding* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/padding.py#L98) 
:   Pads the input tensor using circular padding of the input boundary. 

Tensor values at the beginning of the dimension are used to pad the end,
and values at the end are used to pad the beginning. If negative padding is
applied then the ends of the tensor get removed. 

For *N* -dimensional padding, use [`torch.nn.functional.pad()`](torch.nn.functional.pad.html#torch.nn.functional.pad "torch.nn.functional.pad")  . 

Parameters
: **padding** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  ) – the size of the padding. If is *int* , uses the same
padding in all boundaries. If a 4- *tuple* , uses ( <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
              padding_left
             </mtext>
</mrow>
<annotation encoding="application/x-tex">
             text{padding_left}
            </annotation>
</semantics>
</math> -->padding_left text{padding_left}padding_left  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
              padding_right
             </mtext>
</mrow>
<annotation encoding="application/x-tex">
             text{padding_right}
            </annotation>
</semantics>
</math> -->padding_right text{padding_right}padding_right  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
              padding_top
             </mtext>
</mrow>
<annotation encoding="application/x-tex">
             text{padding_top}
            </annotation>
</semantics>
</math> -->padding_top text{padding_top}padding_top  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
              padding_bottom
             </mtext>
</mrow>
<annotation encoding="application/x-tex">
             text{padding_bottom}
            </annotation>
</semantics>
</math> -->padding_bottom text{padding_bottom}padding_bottom  )
Note that padding size should be less than or equal to the corresponding input dimension.

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

    <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                        padding_top
                       </mtext>
        <mo>
                        +
                       </mo>
        <mtext>
                        padding_bottom
                       </mtext>
        </mrow>
        <annotation encoding="application/x-tex">
                       H_{out} = H_{in} + text{padding_top} + text{padding_bottom}
                      </annotation>
        </semantics>
        </math> -->H o u t = H i n + padding_top + padding_bottom H_{out} = H_{in} + text{padding_top} + text{padding_bottom}H o u t ​ = H in ​ + padding_top + padding_bottom

    <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
        <mtext>
                        padding_left
                       </mtext>
        <mo>
                        +
                       </mo>
        <mtext>
                        padding_right
                       </mtext>
        </mrow>
        <annotation encoding="application/x-tex">
                       W_{out} = W_{in} + text{padding_left} + text{padding_right}
                      </annotation>
        </semantics>
        </math> -->W o u t = W i n + padding_left + padding_right W_{out} = W_{in} + text{padding_left} + text{padding_right}W o u t ​ = W in ​ + padding_left + padding_right

Examples: 

```
>>> m = nn.CircularPad2d(2)
>>> input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
>>> input
tensor([[[[0., 1., 2.],
          [3., 4., 5.],
          [6., 7., 8.]]]])
>>> m(input)
tensor([[[[4., 5., 3., 4., 5., 3., 4.],
          [7., 8., 6., 7., 8., 6., 7.],
          [1., 2., 0., 1., 2., 0., 1.],
          [4., 5., 3., 4., 5., 3., 4.],
          [7., 8., 6., 7., 8., 6., 7.],
          [1., 2., 0., 1., 2., 0., 1.],
          [4., 5., 3., 4., 5., 3., 4.]]]])
>>> # using different paddings for different sides
>>> m = nn.CircularPad2d((1, 1, 2, 0))
>>> m(input)
tensor([[[[5., 3., 4., 5., 3.],
          [8., 6., 7., 8., 6.],
          [2., 0., 1., 2., 0.],
          [5., 3., 4., 5., 3.],
          [8., 6., 7., 8., 6.]]]])

```

