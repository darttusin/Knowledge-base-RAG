FractionalMaxPool2d 
==========================================================================

*class* torch.nn. FractionalMaxPool2d ( *kernel_size*  , *output_size = None*  , *output_ratio = None*  , *return_indices = False*  , *_random_samples = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/pooling.py#L906) 
:   Applies a 2D fractional max pooling over an input signal composed of several input planes. 

Fractional MaxPooling is described in detail in the paper [Fractional MaxPooling](https://arxiv.org/abs/1412.6071)  by Ben Graham 

The max-pooling operation is applied in <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            k
           </mi>
<mi>
            H
           </mi>
<mo>
            ×
           </mo>
<mi>
            k
           </mi>
<mi>
            W
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           kH times kW
          </annotation>
</semantics>
</math> -->k H × k W kH times kWk H × kW  regions by a stochastic
step size determined by the target output size.
The number of output features is equal to the number of input planes. 

Note 

Exactly one of `output_size`  or `output_ratio`  must be defined.

Parameters
:   * **kernel_size** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – the size of the window to take a max over.
Can be a single number k (for a square kernel of k x k) or a tuple *(kh, kw)*
* **output_size** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – the target output size of the image of the form *oH x oW* .
Can be a tuple *(oH, oW)* or a single number oH for a square image *oH x oH* .
Note that we must have <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                k
               </mi>
<mi>
                H
               </mi>
<mo>
                +
               </mo>
<mi>
                o
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
<mo>
                &lt;
               </mo>
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
</mrow>
<annotation encoding="application/x-tex">
               kH + oH - 1 &lt;= H_{in}
              </annotation>
</semantics>
</math> -->k H + o H − 1 < = H i n kH + oH - 1 <= H_{in}k H + oH − 1 <= H in ​  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                k
               </mi>
<mi>
                W
               </mi>
<mo>
                +
               </mo>
<mi>
                o
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
<mo>
                &lt;
               </mo>
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
</mrow>
<annotation encoding="application/x-tex">
               kW + oW - 1 &lt;= W_{in}
              </annotation>
</semantics>
</math> -->k W + o W − 1 < = W i n kW + oW - 1 <= W_{in}kW + o W − 1 <= W in ​

* **output_ratio** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *]* *]*  ) – If one wants to have an output size as a ratio of the input size, this option can be given.
This has to be a number or tuple in the range (0, 1).
Note that we must have <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                k
               </mi>
<mi>
                H
               </mi>
<mo>
                +
               </mo>
<mo stretchy="false">
                (
               </mo>
<mi>
                o
               </mi>
<mi>
                u
               </mi>
<mi>
                t
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
<mi mathvariant="normal">
                _
               </mi>
<mi>
                r
               </mi>
<mi>
                a
               </mi>
<mi>
                t
               </mi>
<mi>
                i
               </mi>
<mi>
                o
               </mi>
<mi mathvariant="normal">
                _
               </mi>
<mi>
                H
               </mi>
<mo>
                ∗
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
<mo stretchy="false">
                )
               </mo>
<mo>
                −
               </mo>
<mn>
                1
               </mn>
<mo>
                &lt;
               </mo>
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
</mrow>
<annotation encoding="application/x-tex">
               kH + (output_ratio_H * H_{in}) - 1 &lt;= H_{in}
              </annotation>
</semantics>
</math> -->k H + ( o u t p u t _ r a t i o _ H ∗ H i n ) − 1 < = H i n kH + (output_ratio_H * H_{in}) - 1 <= H_{in}k H + ( o u tp u t _ r a t i o _ H ∗ H in ​ ) − 1 <= H in ​  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                k
               </mi>
<mi>
                W
               </mi>
<mo>
                +
               </mo>
<mo stretchy="false">
                (
               </mo>
<mi>
                o
               </mi>
<mi>
                u
               </mi>
<mi>
                t
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
<mi mathvariant="normal">
                _
               </mi>
<mi>
                r
               </mi>
<mi>
                a
               </mi>
<mi>
                t
               </mi>
<mi>
                i
               </mi>
<mi>
                o
               </mi>
<mi mathvariant="normal">
                _
               </mi>
<mi>
                W
               </mi>
<mo>
                ∗
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
<mo>
                −
               </mo>
<mn>
                1
               </mn>
<mo>
                &lt;
               </mo>
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
</mrow>
<annotation encoding="application/x-tex">
               kW + (output_ratio_W * W_{in}) - 1 &lt;= W_{in}
              </annotation>
</semantics>
</math> -->k W + ( o u t p u t _ r a t i o _ W ∗ W i n ) − 1 < = W i n kW + (output_ratio_W * W_{in}) - 1 <= W_{in}kW + ( o u tp u t _ r a t i o _ W ∗ W in ​ ) − 1 <= W in ​

* **return_indices** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – if `True`  , will return the indices along with the outputs.
Useful to pass to `nn.MaxUnpool2d()`  . Default: `False`

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
</math> -->( C , H o u t , W o u t ) (C, H_{out}, W_{out})( C , H o u t ​ , W o u t ​ )  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
               (H_{out}, W_{out})=text{output_size}
              </annotation>
</semantics>
</math> -->( H o u t , W o u t ) = output_size (H_{out}, W_{out})=text{output_size}( H o u t ​ , W o u t ​ ) = output_size  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                output_ratio
               </mtext>
<mo>
                ×
               </mo>
<mo stretchy="false">
                (
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
               (H_{out}, W_{out})=text{output_ratio} times (H_{in}, W_{in})
              </annotation>
</semantics>
</math> -->( H o u t , W o u t ) = output_ratio × ( H i n , W i n ) (H_{out}, W_{out})=text{output_ratio} times (H_{in}, W_{in})( H o u t ​ , W o u t ​ ) = output_ratio × ( H in ​ , W in ​ )  .

Examples 

```
>>> # pool of square window of size=3, and target output size 13x12
>>> m = nn.FractionalMaxPool2d(3, output_size=(13, 12))
>>> # pool of square window and target output size being half of input image size
>>> m = nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5))
>>> input = torch.randn(20, 16, 50, 32)
>>> output = m(input)

```

