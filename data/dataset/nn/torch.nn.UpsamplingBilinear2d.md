UpsamplingBilinear2d 
============================================================================

*class* torch.nn. UpsamplingBilinear2d ( *size = None*  , *scale_factor = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/upsampling.py#L245) 
:   Applies a 2D bilinear upsampling to an input signal composed of several input channels. 

To specify the scale, it takes either the `size`  or the `scale_factor`  as it’s constructor argument. 

When `size`  is given, it is the output size of the image *(h, w)* . 

Parameters
:   * **size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* *Tuple* *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *,* *optional*  ) – output spatial sizes
* **scale_factor** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *or* *Tuple* *[* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *]* *,* *optional*  ) – multiplier for
spatial size.

Warning 

This class is deprecated in favor of `interpolate()`  . It is
equivalent to `nn.functional.interpolate(..., mode='bilinear', align_corners=True)`  .

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
</math> -->( N , C , H i n , W i n ) (N, C, H_{in}, W_{in})( N , C , H in ​ , W in ​ )

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
</math> -->( N , C , H o u t , W o u t ) (N, C, H_{out}, W_{out})( N , C , H o u t ​ , W o u t ​ )  where

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
             ×
            </mo>
<mtext>
             scale_factor
            </mtext>
<mo fence="true">
             ⌋
            </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           H_{out} = leftlfloor H_{in} times text{scale_factor} rightrfloor
          </annotation>
</semantics>
</math> -->
H o u t = ⌊ H i n × scale_factor ⌋ H_{out} = leftlfloor H_{in} times text{scale_factor} rightrfloor

H o u t ​ = ⌊ H in ​ × scale_factor ⌋

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
             ×
            </mo>
<mtext>
             scale_factor
            </mtext>
<mo fence="true">
             ⌋
            </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           W_{out} = leftlfloor W_{in} times text{scale_factor} rightrfloor
          </annotation>
</semantics>
</math> -->
W o u t = ⌊ W i n × scale_factor ⌋ W_{out} = leftlfloor W_{in} times text{scale_factor} rightrfloor

W o u t ​ = ⌊ W in ​ × scale_factor ⌋

Examples: 

```
>>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
>>> input
tensor([[[[1., 2.],
          [3., 4.]]]])

>>> m = nn.UpsamplingBilinear2d(scale_factor=2)
>>> m(input)
tensor([[[[1.0000, 1.3333, 1.6667, 2.0000],
          [1.6667, 2.0000, 2.3333, 2.6667],
          [2.3333, 2.6667, 3.0000, 3.3333],
          [3.0000, 3.3333, 3.6667, 4.0000]]]])

```

