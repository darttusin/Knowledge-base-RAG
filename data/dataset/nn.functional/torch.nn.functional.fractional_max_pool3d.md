torch.nn.functional.fractional_max_pool3d 
========================================================================================================================

torch.nn.functional. fractional_max_pool3d ( *input*  , *kernel_size*  , *output_size = None*  , *output_ratio = None*  , *return_indices = False*  , *_random_samples = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_jit_internal.py#L617) 
:   Applies 3D fractional max pooling over an input signal composed of several input planes. 

Fractional MaxPooling is described in detail in the paper [Fractional MaxPooling](http://arxiv.org/abs/1412.6071)  by Ben Graham 

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

Parameters
:   * **kernel_size** – the size of the window to take a max over.
Can be a single number <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                k
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               k
              </annotation>
</semantics>
</math> -->k kk  (for a square kernel of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                k
               </mi>
<mo>
                ×
               </mo>
<mi>
                k
               </mi>
<mo>
                ×
               </mo>
<mi>
                k
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               k times k times k
              </annotation>
</semantics>
</math> -->k × k × k k times k times kk × k × k  )
or a tuple *(kT, kH, kW)*

* **output_size** – the target output size of the form <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                o
               </mi>
<mi>
                T
               </mi>
<mo>
                ×
               </mo>
<mi>
                o
               </mi>
<mi>
                H
               </mi>
<mo>
                ×
               </mo>
<mi>
                o
               </mi>
<mi>
                W
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               oT times oH times oW
              </annotation>
</semantics>
</math> -->o T × o H × o W oT times oH times oWo T × oH × o W  .
Can be a tuple *(oT, oH, oW)* or a single number <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                o
               </mi>
<mi>
                H
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               oH
              </annotation>
</semantics>
</math> -->o H oHoH  for a cubic output <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                o
               </mi>
<mi>
                H
               </mi>
<mo>
                ×
               </mo>
<mi>
                o
               </mi>
<mi>
                H
               </mi>
<mo>
                ×
               </mo>
<mi>
                o
               </mi>
<mi>
                H
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               oH times oH times oH
              </annotation>
</semantics>
</math> -->o H × o H × o H oH times oH times oHoH × oH × oH

* **output_ratio** – If one wants to have an output size as a ratio of the input size, this option can be given.
This has to be a number or tuple in the range (0, 1)
* **return_indices** – if `True`  , will return the indices along with the outputs.
Useful to pass to [`max_unpool3d()`](torch.nn.functional.max_unpool3d.html#torch.nn.functional.max_unpool3d "torch.nn.functional.max_unpool3d")  .

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

Examples::
:   ```
>>> input = torch.randn(20, 16, 50, 32, 16)
>>> # pool of cubic window of size=3, and target output size 13x12x11
>>> F.fractional_max_pool3d(input, 3, output_size=(13, 12, 11))
>>> # pool of cubic window and target output size being half of input size
>>> F.fractional_max_pool3d(input, 3, output_ratio=(0.5, 0.5, 0.5))

```

