AvgPool1d 
======================================================

*class* torch.nn. AvgPool1d ( *kernel_size*  , *stride = None*  , *padding = 0*  , *ceil_mode = False*  , *count_include_pad = True* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/pooling.py#L591) 
:   Applies a 1D average pooling over an input signal composed of several input planes. 

In the simplest case, the output value of the layer with input size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
            L
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (N, C, L)
          </annotation>
</semantics>
</math> -->( N , C , L ) (N, C, L)( N , C , L )  ,
output <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N , C , L o u t ) (N, C, L_{out})( N , C , L o u t ​ )  and `kernel_size` <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->k kk  can be precisely described as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            out
           </mtext>
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             N
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
             C
            </mi>
<mi>
             j
            </mi>
</msub>
<mo separator="true">
            ,
           </mo>
<mi>
            l
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mfrac>
<mn>
             1
            </mn>
<mi>
             k
            </mi>
</mfrac>
<munderover>
<mo>
             ∑
            </mo>
<mrow>
<mi>
              m
             </mi>
<mo>
              =
             </mo>
<mn>
              0
             </mn>
</mrow>
<mrow>
<mi>
              k
             </mi>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
</mrow>
</munderover>
<mtext>
            input
           </mtext>
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             N
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
             C
            </mi>
<mi>
             j
            </mi>
</msub>
<mo separator="true">
            ,
           </mo>
<mtext>
            stride
           </mtext>
<mo>
            ×
           </mo>
<mi>
            l
           </mi>
<mo>
            +
           </mo>
<mi>
            m
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{out}(N_i, C_j, l) = frac{1}{k} sum_{m=0}^{k-1}
                       text{input}(N_i, C_j, text{stride} times l + m)
          </annotation>
</semantics>
</math> -->
out ( N i , C j , l ) = 1 k ∑ m = 0 k − 1 input ( N i , C j , stride × l + m ) text{out}(N_i, C_j, l) = frac{1}{k} sum_{m=0}^{k-1}
 text{input}(N_i, C_j, text{stride} times l + m)

out ( N i ​ , C j ​ , l ) = k 1 ​ m = 0 ∑ k − 1 ​ input ( N i ​ , C j ​ , stride × l + m )

If `padding`  is non-zero, then the input is implicitly zero-padded on both sides
for `padding`  number of points. 

Note 

When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
or the input. Sliding windows that would start in the right padded region are ignored.

Note 

pad should be at most half of effective kernel size.

The parameters `kernel_size`  , `stride`  , `padding`  can each be
an `int`  or a one-element tuple. 

Parameters
:   * **kernel_size** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – the size of the window
* **stride** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – the stride of the window. Default value is `kernel_size`
* **padding** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – implicit zero padding to be added on both sides
* **ceil_mode** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – when True, will use *ceil* instead of *floor* to compute the output shape
* **count_include_pad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – when True, will include the zero-padding in the averaging calculation

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
                       +
                      </mo>
    <mn>
                       2
                      </mn>
    <mo>
                       ×
                      </mo>
    <mtext>
                       padding
                      </mtext>
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
                   L_{out} = leftlfloor frac{L_{in} +
    2 times text{padding} - text{kernel_size}}{text{stride}} + 1rightrfloor
                  </annotation>
    </semantics>
    </math> -->
    L o u t = ⌊ L i n + 2 × padding − kernel_size stride + 1 ⌋ L_{out} = leftlfloor frac{L_{in} +
    2 times text{padding} - text{kernel_size}}{text{stride}} + 1rightrfloor

    L o u t ​ = ⌊ stride L in ​ + 2 × padding − kernel_size ​ + 1 ⌋

    Per the note above, if `ceil_mode`  is True and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
        <semantics>
        <mrow>
        <mo stretchy="false">
                        (
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
        <mo>
                        −
                       </mo>
        <mn>
                        1
                       </mn>
        <mo stretchy="false">
                        )
                       </mo>
        <mo>
                        ×
                       </mo>
        <mtext>
                        stride
                       </mtext>
        <mo>
                        ≥
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
        <mo>
                        +
                       </mo>
        <mtext>
                        padding
                       </mtext>
        </mrow>
        <annotation encoding="application/x-tex">
                       (L_{out} - 1) times text{stride} geq L_{in}

        + text{padding}
                      </annotation>
        </semantics>
        </math> -->( L o u t − 1 ) × stride ≥ L i n + padding (L_{out} - 1) times text{stride} geq L_{in}
        + text{padding}( L o u t ​ − 1 ) × stride ≥ L in ​ + padding  , we skip the last window as it would start in the right padded region, resulting in <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
        </mrow>
        <annotation encoding="application/x-tex">
                       L_{out}
                      </annotation>
        </semantics>
        </math> -->L o u t L_{out}L o u t ​  being reduced by one.

Examples: 

```
>>> # pool with window of size=3, stride=2
>>> m = nn.AvgPool1d(3, stride=2)
>>> m(torch.tensor([[[1., 2, 3, 4, 5, 6, 7]]]))
tensor([[[2., 4., 6.]]])

```

