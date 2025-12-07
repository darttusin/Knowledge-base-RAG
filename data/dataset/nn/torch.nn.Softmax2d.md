Softmax2d 
======================================================

*class* torch.nn. Softmax2d ( ** args*  , *** kwargs* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L1685) 
:   Applies SoftMax over features to each spatial location. 

When given an image of `Channels x Height x Width`  , it will
apply *Softmax* to each location <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mi>
            C
           </mi>
<mi>
            h
           </mi>
<mi>
            a
           </mi>
<mi>
            n
           </mi>
<mi>
            n
           </mi>
<mi>
            e
           </mi>
<mi>
            l
           </mi>
<mi>
            s
           </mi>
<mo separator="true">
            ,
           </mo>
<msub>
<mi>
             h
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
             w
            </mi>
<mi>
             j
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (Channels, h_i, w_j)
          </annotation>
</semantics>
</math> -->( C h a n n e l s , h i , w j ) (Channels, h_i, w_j)( C hann e l s , h i ​ , w j ​ ) 

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
</math> -->( N , C , H , W ) (N, C, H, W)( N , C , H , W )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
               (C, H, W)
              </annotation>
</semantics>
</math> -->( C , H , W ) (C, H, W)( C , H , W )  .

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
</math> -->( N , C , H , W ) (N, C, H, W)( N , C , H , W )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
               (C, H, W)
              </annotation>
</semantics>
</math> -->( C , H , W ) (C, H, W)( C , H , W )  (same shape as input)

Returns
:   a Tensor of the same dimension and shape as the input with
values in the range [0, 1]

Return type
:   None

Examples: 

```
>>> m = nn.Softmax2d()
>>> # you softmax over the 2nd dimension
>>> input = torch.randn(2, 3, 12, 13)
>>> output = m(input)

```

