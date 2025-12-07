PixelShuffle 
============================================================

*class* torch.nn. PixelShuffle ( *upscale_factor* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/pixelshuffle.py#L10) 
:   Rearrange elements in a tensor according to an upscaling factor. 

Rearranges elements in a tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mo>
            ∗
           </mo>
<mo separator="true">
            ,
           </mo>
<mi>
            C
           </mi>
<mo>
            ×
           </mo>
<msup>
<mi>
             r
            </mi>
<mn>
             2
            </mn>
</msup>
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
           (*, C times r^2, H, W)
          </annotation>
</semantics>
</math> -->( ∗ , C × r 2 , H , W ) (*, C times r^2, H, W)( ∗ , C × r 2 , H , W )  to a tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mo>
            ∗
           </mo>
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
<mo>
            ×
           </mo>
<mi>
            r
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            W
           </mi>
<mo>
            ×
           </mo>
<mi>
            r
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (*, C, H times r, W times r)
          </annotation>
</semantics>
</math> -->( ∗ , C , H × r , W × r ) (*, C, H times r, W times r)( ∗ , C , H × r , W × r )  , where r is an upscale factor. 

This is useful for implementing efficient sub-pixel convolution
with a stride of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
            1
           </mn>
<mi mathvariant="normal">
            /
           </mi>
<mi>
            r
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           1/r
          </annotation>
</semantics>
</math> -->1 / r 1/r1/ r  . 

See the paper: [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)  by Shi et al. (2016) for more details. 

Parameters
: **upscale_factor** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – factor to increase spatial resolution by

Shape:
:   * Input: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 C
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
               (*, C_{in}, H_{in}, W_{in})
              </annotation>
</semantics>
</math> -->( ∗ , C i n , H i n , W i n ) (*, C_{in}, H_{in}, W_{in})( ∗ , C in ​ , H in ​ , W in ​ )  , where * is zero or more batch dimensions

* Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 C
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
               (*, C_{out}, H_{out}, W_{out})
              </annotation>
</semantics>
</math> -->( ∗ , C o u t , H o u t , W o u t ) (*, C_{out}, H_{out}, W_{out})( ∗ , C o u t ​ , H o u t ​ , W o u t ​ )  , where

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             C
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
             C
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
            ÷
           </mo>
<msup>
<mtext>
             upscale_factor
            </mtext>
<mn>
             2
            </mn>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           C_{out} = C_{in} div text{upscale_factor}^2
          </annotation>
</semantics>
</math> -->
C o u t = C i n ÷ upscale_factor 2 C_{out} = C_{in} div text{upscale_factor}^2

C o u t ​ = C in ​ ÷ upscale_factor 2

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
            upscale_factor
           </mtext>
</mrow>
<annotation encoding="application/x-tex">
           H_{out} = H_{in} times text{upscale_factor}
          </annotation>
</semantics>
</math> -->
H o u t = H i n × upscale_factor H_{out} = H_{in} times text{upscale_factor}

H o u t ​ = H in ​ × upscale_factor

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
            upscale_factor
           </mtext>
</mrow>
<annotation encoding="application/x-tex">
           W_{out} = W_{in} times text{upscale_factor}
          </annotation>
</semantics>
</math> -->
W o u t = W i n × upscale_factor W_{out} = W_{in} times text{upscale_factor}

W o u t ​ = W in ​ × upscale_factor

Examples: 

```
>>> pixel_shuffle = nn.PixelShuffle(3)
>>> input = torch.randn(1, 9, 4, 4)
>>> output = pixel_shuffle(input)
>>> print(output.size())
torch.Size([1, 1, 12, 12])

```

