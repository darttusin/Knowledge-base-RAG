FractionalMaxPool3d 
==========================================================================

*class* torch.nn. FractionalMaxPool3d ( *kernel_size*  , *output_size = None*  , *output_ratio = None*  , *return_indices = False*  , *_random_samples = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/pooling.py#L995) 
:   Applies a 3D fractional max pooling over an input signal composed of several input planes. 

Fractional MaxPooling is described in detail in the paper [Fractional MaxPooling](https://arxiv.org/abs/1412.6071)  by Ben Graham 

The max-pooling operation is applied in <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            k
           </mi>
<mi>
            T
           </mi>
<mo>
            ×
           </mo>
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
           kT times kH times kW
          </annotation>
</semantics>
</math> -->k T × k H × k W kT times kH times kWk T × k H × kW  regions by a stochastic
step size determined by the target output size.
The number of output features is equal to the number of input planes. 

Note 

Exactly one of `output_size`  or `output_ratio`  must be defined.

Parameters
:   * **kernel_size** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – the size of the window to take a max over.
Can be a single number k (for a square kernel of k x k x k) or a tuple *(kt x kh x kw)*
* **output_size** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – the target output size of the image of the form *oT x oH x oW* .
Can be a tuple *(oT, oH, oW)* or a single number oH for a square image *oH x oH x oH*
* **output_ratio** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *]* *]*  ) – If one wants to have an output size as a ratio of the input size, this option can be given.
This has to be a number or tuple in the range (0, 1)
* **return_indices** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – if `True`  , will return the indices along with the outputs.
Useful to pass to `nn.MaxUnpool3d()`  . Default: `False`

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
                 T
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
               (N, C, T_{in}, H_{in}, W_{in})
              </annotation>
</semantics>
</math> -->( N , C , T i n , H i n , W i n ) (N, C, T_{in}, H_{in}, W_{in})( N , C , T in ​ , H in ​ , W in ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                 T
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
               (C, T_{in}, H_{in}, W_{in})
              </annotation>
</semantics>
</math> -->( C , T i n , H i n , W i n ) (C, T_{in}, H_{in}, W_{in})( C , T in ​ , H in ​ , W in ​ )  .

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
                 T
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
               (N, C, T_{out}, H_{out}, W_{out})
              </annotation>
</semantics>
</math> -->( N , C , T o u t , H o u t , W o u t ) (N, C, T_{out}, H_{out}, W_{out})( N , C , T o u t ​ , H o u t ​ , W o u t ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                 T
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
               (C, T_{out}, H_{out}, W_{out})
              </annotation>
</semantics>
</math> -->( C , T o u t , H o u t , W o u t ) (C, T_{out}, H_{out}, W_{out})( C , T o u t ​ , H o u t ​ , W o u t ​ )  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 T
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
               (T_{out}, H_{out}, W_{out})=text{output_size}
              </annotation>
</semantics>
</math> -->( T o u t , H o u t , W o u t ) = output_size (T_{out}, H_{out}, W_{out})=text{output_size}( T o u t ​ , H o u t ​ , W o u t ​ ) = output_size  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 T
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
                 T
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
               (T_{out}, H_{out}, W_{out})=text{output_ratio} times (T_{in}, H_{in}, W_{in})
              </annotation>
</semantics>
</math> -->( T o u t , H o u t , W o u t ) = output_ratio × ( T i n , H i n , W i n ) (T_{out}, H_{out}, W_{out})=text{output_ratio} times (T_{in}, H_{in}, W_{in})( T o u t ​ , H o u t ​ , W o u t ​ ) = output_ratio × ( T in ​ , H in ​ , W in ​ )

Examples 

```
>>> # pool of cubic window of size=3, and target output size 13x12x11
>>> m = nn.FractionalMaxPool3d(3, output_size=(13, 12, 11))
>>> # pool of cubic window and target output size being half of input size
>>> m = nn.FractionalMaxPool3d(3, output_ratio=(0.5, 0.5, 0.5))
>>> input = torch.randn(20, 16, 50, 32, 16)
>>> output = m(input)

```

