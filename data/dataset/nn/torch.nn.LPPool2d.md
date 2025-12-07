LPPool2d 
====================================================

*class* torch.nn. LPPool2d ( *norm_type*  , *kernel_size*  , *stride = None*  , *ceil_mode = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/pooling.py#L1152) 
:   Applies a 2D power-average pooling over an input signal composed of several input planes. 

On each window, the function computed is: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            f
           </mi>
<mo stretchy="false">
            (
           </mo>
<mi>
            X
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mroot>
<mrow>
<munder>
<mo>
               ∑
              </mo>
<mrow>
<mi>
                x
               </mi>
<mo>
                ∈
               </mo>
<mi>
                X
               </mi>
</mrow>
</munder>
<msup>
<mi>
               x
              </mi>
<mi>
               p
              </mi>
</msup>
</mrow>
<mi>
             p
            </mi>
</mroot>
</mrow>
<annotation encoding="application/x-tex">
           f(X) = sqrt[p]{sum_{x in X} x^{p}}
          </annotation>
</semantics>
</math> -->
f ( X ) = ∑ x ∈ X x p p f(X) = sqrt[p]{sum_{x in X} x^{p}}

f ( X ) = p ​ x ∈ X ∑ ​ x p ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjMuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMzI0MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNNDczLDI3OTMKYzMzOS4zLC0xNzk5LjMsNTA5LjMsLTI3MDAsNTEwLC0yNzAyIGwwIC0wCmMzLjMsLTcuMyw5LjMsLTExLDE4LC0xMSBINDAwMDAwdjQwSDEwMTcuNwpzLTkwLjUsNDc4LC0yNzYuMiwxNDY2Yy0xODUuNyw5ODgsLTI3OS41LDE0ODMsLTI4MS41LDE0ODVjLTIsNiwtMTAsOSwtMjQsOQpjLTgsMCwtMTIsLTAuNywtMTIsLTJjMCwtMS4zLC01LjMsLTMyLC0xNiwtOTJjLTUwLjcsLTI5My4zLC0xMTkuNywtNjkzLjMsLTIwNywtMTIwMApjMCwtMS4zLC01LjMsOC43LC0xNiwzMGMtMTAuNywyMS4zLC0yMS4zLDQyLjcsLTMyLDY0cy0xNiwzMywtMTYsMzNzLTI2LC0yNiwtMjYsLTI2CnM3NiwtMTUzLDc2LC0xNTNzNzcsLTE1MSw3NywtMTUxYzAuNywwLjcsMzUuNywyMDIsMTA1LDYwNGM2Ny4zLDQwMC43LDEwMiw2MDIuNywxMDQsCjYwNnpNMTAwMSA4MGg0MDAwMDB2NDBIMTAxNy43eiI+CjwvcGF0aD4KPC9zdmc+)​

* At p = <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
              ∞
             </mi>
</mrow>
<annotation encoding="application/x-tex">
             infty
            </annotation>
</semantics>
</math> -->∞ infty∞  , one gets Max Pooling

* At p = 1, one gets Sum Pooling (which is proportional to average pooling)

The parameters `kernel_size`  , `stride`  can either be: 

> * a single `int`  – in which case the same value is used for the height and width dimension
> * a `tuple`  of two ints – in which case, the first *int* is used for the height dimension,
> and the second *int* for the width dimension

Note 

If the sum to the power of *p* is zero, the gradient of this function is
not defined. This implementation will set the gradient to zero in this case.

Parameters
:   * **kernel_size** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – the size of the window
* **stride** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – the stride of the window. Default value is `kernel_size`
* **ceil_mode** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – when True, will use *ceil* instead of *floor* to compute the output shape

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
</math> -->( C , H o u t , W o u t ) (C, H_{out}, W_{out})( C , H o u t ​ , W o u t ​ )  , where

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
    <mfrac>
    <mrow>
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
                       −
                      </mo>
    <mtext>
                       kernel_size
                      </mtext>
    <mo stretchy="false">
                       [
                      </mo>
    <mn>
                       0
                      </mn>
    <mo stretchy="false">
                       ]
                      </mo>
    </mrow>
    <mrow>
    <mtext>
                       stride
                      </mtext>
    <mo stretchy="false">
                       [
                      </mo>
    <mn>
                       0
                      </mn>
    <mo stretchy="false">
                       ]
                      </mo>
    </mrow>
    </mfrac>
    <mo>
                     +
                    </mo>
    <mn>
                     1
                    </mn>
    <mo fence="true">
                     ⌋
                    </mo>
    </mrow>
    </mrow>
    <annotation encoding="application/x-tex">
                   H_{out} = leftlfloorfrac{H_{in} - text{kernel_size}[0]}{text{stride}[0]} + 1rightrfloor
                  </annotation>
    </semantics>
    </math> -->
    H o u t = ⌊ H i n − kernel_size [ 0 ] stride [ 0 ] + 1 ⌋ H_{out} = leftlfloorfrac{H_{in} - text{kernel_size}[0]}{text{stride}[0]} + 1rightrfloor

    H o u t ​ = ⌊ stride [ 0 ] H in ​ − kernel_size [ 0 ] ​ + 1 ⌋

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
        <mfrac>
        <mrow>
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
                           −
                          </mo>
        <mtext>
                           kernel_size
                          </mtext>
        <mo stretchy="false">
                           [
                          </mo>
        <mn>
                           1
                          </mn>
        <mo stretchy="false">
                           ]
                          </mo>
        </mrow>
        <mrow>
        <mtext>
                           stride
                          </mtext>
        <mo stretchy="false">
                           [
                          </mo>
        <mn>
                           1
                          </mn>
        <mo stretchy="false">
                           ]
                          </mo>
        </mrow>
        </mfrac>
        <mo>
                         +
                        </mo>
        <mn>
                         1
                        </mn>
        <mo fence="true">
                         ⌋
                        </mo>
        </mrow>
        </mrow>
        <annotation encoding="application/x-tex">
                       W_{out} = leftlfloorfrac{W_{in} - text{kernel_size}[1]}{text{stride}[1]} + 1rightrfloor
                      </annotation>
        </semantics>
        </math> -->
        W o u t = ⌊ W i n − kernel_size [ 1 ] stride [ 1 ] + 1 ⌋ W_{out} = leftlfloorfrac{W_{in} - text{kernel_size}[1]}{text{stride}[1]} + 1rightrfloor

    W o u t ​ = ⌊ stride [ 1 ] W in ​ − kernel_size [ 1 ] ​ + 1 ⌋

Examples: 

```
>>> # power-2 pool of square window of size=3, stride=2
>>> m = nn.LPPool2d(2, 3, stride=2)
>>> # pool of non-square window of power 1.2
>>> m = nn.LPPool2d(1.2, (3, 2), stride=(2, 1))
>>> input = torch.randn(20, 16, 50, 32)
>>> output = m(input)

```

