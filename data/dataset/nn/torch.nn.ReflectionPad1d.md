ReflectionPad1d 
==================================================================

*class* torch.nn. ReflectionPad1d ( *padding* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/padding.py#L377) 
:   Pads the input tensor using the reflection of the input boundary. 

For *N* -dimensional padding, use [`torch.nn.functional.pad()`](torch.nn.functional.pad.html#torch.nn.functional.pad "torch.nn.functional.pad")  . 

Parameters
: **padding** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  ) – the size of the padding. If is *int* , uses the same
padding in all boundaries. If a 2- *tuple* , uses
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
</math> -->padding_right text{padding_right}padding_right  )
Note that padding size should be less than the corresponding input dimension.

Shape:
:   * Input: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
               (C, W_{in})
              </annotation>
</semantics>
</math> -->( C , W i n ) (C, W_{in})( C , W in ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
               (N, C, W_{in})
              </annotation>
</semantics>
</math> -->( N , C , W i n ) (N, C, W_{in})( N , C , W in ​ )  .

* Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
               (C, W_{out})
              </annotation>
</semantics>
</math> -->( C , W o u t ) (C, W_{out})( C , W o u t ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
               (N, C, W_{out})
              </annotation>
</semantics>
</math> -->( N , C , W o u t ) (N, C, W_{out})( N , C , W o u t ​ )  , where

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
>>> m = nn.ReflectionPad1d(2)
>>> input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
>>> input
tensor([[[0., 1., 2., 3.],
         [4., 5., 6., 7.]]])
>>> m(input)
tensor([[[2., 1., 0., 1., 2., 3., 2., 1.],
         [6., 5., 4., 5., 6., 7., 6., 5.]]])
>>> # using different paddings for different sides
>>> m = nn.ReflectionPad1d((3, 1))
>>> m(input)
tensor([[[3., 2., 1., 0., 1., 2., 3., 2.],
         [7., 6., 5., 4., 5., 6., 7., 6.]]])

```

