ConstantPad3d 
==============================================================

*class* torch.nn. ConstantPad3d ( *padding*  , *value* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/padding.py#L326) 
:   Pads the input tensor boundaries with a constant value. 

For *N* -dimensional padding, use [`torch.nn.functional.pad()`](torch.nn.functional.pad.html#torch.nn.functional.pad "torch.nn.functional.pad")  . 

Parameters
: **padding** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  ) – the size of the padding. If is *int* , uses the same
padding in all boundaries. If a 6- *tuple* , uses
( <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->padding_bottom text{padding_bottom}padding_bottom  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
              padding_front
             </mtext>
</mrow>
<annotation encoding="application/x-tex">
             text{padding_front}
            </annotation>
</semantics>
</math> -->padding_front text{padding_front}padding_front  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
              padding_back
             </mtext>
</mrow>
<annotation encoding="application/x-tex">
             text{padding_back}
            </annotation>
</semantics>
</math> -->padding_back text{padding_back}padding_back  )

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

    <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                        padding_front
                       </mtext>
        <mo>
                        +
                       </mo>
        <mtext>
                        padding_back
                       </mtext>
        </mrow>
        <annotation encoding="application/x-tex">
                       D_{out} = D_{in} + text{padding_front} + text{padding_back}
                      </annotation>
        </semantics>
        </math> -->D o u t = D i n + padding_front + padding_back D_{out} = D_{in} + text{padding_front} + text{padding_back}D o u t ​ = D in ​ + padding_front + padding_back

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
>>> m = nn.ConstantPad3d(3, 3.5)
>>> input = torch.randn(16, 3, 10, 20, 30)
>>> output = m(input)
>>> # using different paddings for different sides
>>> m = nn.ConstantPad3d((3, 3, 6, 6, 0, 1), 3.5)
>>> output = m(input)

```

