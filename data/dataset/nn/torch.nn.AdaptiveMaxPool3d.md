AdaptiveMaxPool3d 
======================================================================

*class* torch.nn. AdaptiveMaxPool3d ( *output_size*  , *return_indices = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/pooling.py#L1357) 
:   Applies a 3D adaptive max pooling over an input signal composed of several input planes. 

The output is of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
            ×
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
            ×
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
</mrow>
<annotation encoding="application/x-tex">
           D_{out} times H_{out} times W_{out}
          </annotation>
</semantics>
</math> -->D o u t × H o u t × W o u t D_{out} times H_{out} times W_{out}D o u t ​ × H o u t ​ × W o u t ​  , for any input size.
The number of output features is equal to the number of input planes. 

Parameters
:   * **output_size** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *None* *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *,* [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *,* [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]* *]*  ) – the target output size of the image of the form <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                ×
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
                ×
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
</mrow>
<annotation encoding="application/x-tex">
               D_{out} times H_{out} times W_{out}
              </annotation>
</semantics>
</math> -->D o u t × H o u t × W o u t D_{out} times H_{out} times W_{out}D o u t ​ × H o u t ​ × W o u t ​  .
Can be a tuple <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
               (D_{out}, H_{out}, W_{out})
              </annotation>
</semantics>
</math> -->( D o u t , H o u t , W o u t ) (D_{out}, H_{out}, W_{out})( D o u t ​ , H o u t ​ , W o u t ​ )  or a single <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->D o u t D_{out}D o u t ​  for a cube <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                ×
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
                ×
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
</mrow>
<annotation encoding="application/x-tex">
               D_{out} times D_{out} times D_{out}
              </annotation>
</semantics>
</math> -->D o u t × D o u t × D o u t D_{out} times D_{out} times D_{out}D o u t ​ × D o u t ​ × D o u t ​  . <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->D o u t D_{out}D o u t ​  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->H o u t H_{out}H o u t ​  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->W o u t W_{out}W o u t ​  can be either a `int`  , or `None`  which means the size will be the same as that of the input.

* **return_indices** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – if `True`  , will return the indices along with the outputs.
Useful to pass to nn.MaxUnpool3d. Default: `False`

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
</math> -->( C , D o u t , H o u t , W o u t ) (C, D_{out}, H_{out}, W_{out})( C , D o u t ​ , H o u t ​ , W o u t ​ )  ,
where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mo>
                =
               </mo>
<mtext>
                output_size
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               (D_{out}, H_{out}, W_{out})=text{output_size}
              </annotation>
</semantics>
</math> -->( D o u t , H o u t , W o u t ) = output_size (D_{out}, H_{out}, W_{out})=text{output_size}( D o u t ​ , H o u t ​ , W o u t ​ ) = output_size  .

Examples 

```
>>> # target output size of 5x7x9
>>> m = nn.AdaptiveMaxPool3d((5, 7, 9))
>>> input = torch.randn(1, 64, 8, 9, 10)
>>> output = m(input)
>>> # target output size of 7x7x7 (cube)
>>> m = nn.AdaptiveMaxPool3d(7)
>>> input = torch.randn(1, 64, 10, 9, 8)
>>> output = m(input)
>>> # target output size of 7x9x8
>>> m = nn.AdaptiveMaxPool3d((7, None, None))
>>> input = torch.randn(1, 64, 10, 9, 8)
>>> output = m(input)

```

