LPPool1d 
====================================================

*class* torch.nn. LPPool1d ( *norm_type*  , *kernel_size*  , *stride = None*  , *ceil_mode = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/pooling.py#L1110) 
:   Applies a 1D power-average pooling over an input signal composed of several input planes. 

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

* At p = 1, one gets Sum Pooling (which is proportional to Average Pooling)

Note 

If the sum to the power of *p* is zero, the gradient of this function is
not defined. This implementation will set the gradient to zero in this case.

Parameters
:   * **kernel_size** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – a single int, the size of the window
* **stride** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – a single int, the stride of the window. Default value is `kernel_size`
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
                 L
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
               (N, C, L_{in})
              </annotation>
</semantics>
</math> -->( N , C , L i n ) (N, C, L_{in})( N , C , L in ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                 L
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
               (C, L_{in})
              </annotation>
</semantics>
</math> -->( C , L i n ) (C, L_{in})( C , L in ​ )  .

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
                 L
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
               (N, C, L_{out})
              </annotation>
</semantics>
</math> -->( N , C , L o u t ) (N, C, L_{out})( N , C , L o u t ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                 L
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
               (C, L_{out})
              </annotation>
</semantics>
</math> -->( C , L o u t ) (C, L_{out})( C , L o u t ​ )  , where

    <!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
    <semantics>
    <mrow>
    <msub>
    <mi>
                     L
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
                        L
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
    </mrow>
    <mtext>
                      stride
                     </mtext>
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
                   L_{out} = leftlfloorfrac{L_{in} - text{kernel_size}}{text{stride}} + 1rightrfloor
                  </annotation>
    </semantics>
    </math> -->
    L o u t = ⌊ L i n − kernel_size stride + 1 ⌋ L_{out} = leftlfloorfrac{L_{in} - text{kernel_size}}{text{stride}} + 1rightrfloor

    L o u t ​ = ⌊ stride L in ​ − kernel_size ​ + 1 ⌋

Examples::
:   ```
>>> # power-2 pool of window of length 3, with stride 2.
>>> m = nn.LPPool1d(2, 3, stride=2)
>>> input = torch.randn(20, 16, 50)
>>> output = m(input)

```

