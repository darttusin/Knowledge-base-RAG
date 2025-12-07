ZeroPad1d 
======================================================

*class* torch.nn. ZeroPad1d ( *padding* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/padding.py#L675) 
:   Pads the input tensor boundaries with zero. 

For *N* -dimensional padding, use [`torch.nn.functional.pad()`](torch.nn.functional.pad.html#torch.nn.functional.pad "torch.nn.functional.pad")  . 

Parameters
: **padding** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  ) – the size of the padding. If is *int* , uses the same
padding in both boundaries. If a 2- *tuple* , uses
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
>>> m = nn.ZeroPad1d(2)
>>> input = torch.randn(1, 2, 4)
>>> input
tensor([[[-1.0491, -0.7152, -0.0749,  0.8530],
         [-1.3287,  1.8966,  0.1466, -0.2771]]])
>>> m(input)
tensor([[[ 0.0000,  0.0000, -1.0491, -0.7152, -0.0749,  0.8530,  0.0000,
           0.0000],
         [ 0.0000,  0.0000, -1.3287,  1.8966,  0.1466, -0.2771,  0.0000,
           0.0000]]])
>>> m = nn.ZeroPad1d(2)
>>> input = torch.randn(1, 2, 3)
>>> input
tensor([[[ 1.6616,  1.4523, -1.1255],
         [-3.6372,  0.1182, -1.8652]]])
>>> m(input)
tensor([[[ 0.0000,  0.0000,  1.6616,  1.4523, -1.1255,  0.0000,  0.0000],
         [ 0.0000,  0.0000, -3.6372,  0.1182, -1.8652,  0.0000,  0.0000]]])
>>> # using different paddings for different sides
>>> m = nn.ZeroPad1d((3, 1))
>>> m(input)
tensor([[[ 0.0000,  0.0000,  0.0000,  1.6616,  1.4523, -1.1255,  0.0000],
         [ 0.0000,  0.0000,  0.0000, -3.6372,  0.1182, -1.8652,  0.0000]]])

```

