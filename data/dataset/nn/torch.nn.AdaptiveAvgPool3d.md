AdaptiveAvgPool3d 
======================================================================

*class* torch.nn. AdaptiveAvgPool3d ( *output_size* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/pooling.py#L1478) 
:   Applies a 3D adaptive average pooling over an input signal composed of several input planes. 

The output is of size D x H x W, for any input size.
The number of output features is equal to the number of input planes. 

Parameters
: **output_size** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *None* *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *,* [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *,* [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]* *]*  ) – the target output size of the form D x H x W.
Can be a tuple (D, H, W) or a single number D for a cube D x D x D.
D, H and W can be either a `int`  , or `None`  which means the size will
be the same as that of the input.

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
                 S
                </mi>
<mn>
                 0
                </mn>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 S
                </mi>
<mn>
                 1
                </mn>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 S
                </mi>
<mn>
                 2
                </mn>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, S_{0}, S_{1}, S_{2})
              </annotation>
</semantics>
</math> -->( N , C , S 0 , S 1 , S 2 ) (N, C, S_{0}, S_{1}, S_{2})( N , C , S 0 ​ , S 1 ​ , S 2 ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                 S
                </mi>
<mn>
                 0
                </mn>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 S
                </mi>
<mn>
                 1
                </mn>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 S
                </mi>
<mn>
                 2
                </mn>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (C, S_{0}, S_{1}, S_{2})
              </annotation>
</semantics>
</math> -->( C , S 0 , S 1 , S 2 ) (C, S_{0}, S_{1}, S_{2})( C , S 0 ​ , S 1 ​ , S 2 ​ )  ,
where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                S
               </mi>
<mo>
                =
               </mo>
<mtext>
                output_size
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               S=text{output_size}
              </annotation>
</semantics>
</math> -->S = output_size S=text{output_size}S = output_size  .

Examples 

```
>>> # target output size of 5x7x9
>>> m = nn.AdaptiveAvgPool3d((5, 7, 9))
>>> input = torch.randn(1, 64, 8, 9, 10)
>>> output = m(input)
>>> # target output size of 7x7x7 (cube)
>>> m = nn.AdaptiveAvgPool3d(7)
>>> input = torch.randn(1, 64, 10, 9, 8)
>>> output = m(input)
>>> # target output size of 7x9x8
>>> m = nn.AdaptiveAvgPool3d((7, None, None))
>>> input = torch.randn(1, 64, 10, 9, 8)
>>> output = m(input)

```

