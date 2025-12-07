MaxPool1d 
======================================================

*class* torch.nn. MaxPool1d ( *kernel_size*  , *stride = None*  , *padding = 0*  , *dilation = 1*  , *return_indices = False*  , *ceil_mode = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/pooling.py#L81) 
:   Applies a 1D max pooling over an input signal composed of several input planes. 

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
</math> -->( N , C , L ) (N, C, L)( N , C , L )  and output <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N , C , L o u t ) (N, C, L_{out})( N , C , L o u t ​ )  can be precisely described as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
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
            k
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<munder>
<mrow>
<mi>
              max
             </mi>
<mo>
              ⁡
             </mo>
</mrow>
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
<mo separator="true">
              ,
             </mo>
<mo>
              …
             </mo>
<mo separator="true">
              ,
             </mo>
<mtext>
              kernel_size
             </mtext>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
</mrow>
</munder>
<mi>
            i
           </mi>
<mi>
            n
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
            s
           </mi>
<mi>
            t
           </mi>
<mi>
            r
           </mi>
<mi>
            i
           </mi>
<mi>
            d
           </mi>
<mi>
            e
           </mi>
<mo>
            ×
           </mo>
<mi>
            k
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
           out(N_i, C_j, k) = max_{m=0, ldots, text{kernel_size} - 1}
        input(N_i, C_j, stride times k + m)
          </annotation>
</semantics>
</math> -->
o u t ( N i , C j , k ) = max ⁡ m = 0 , … , kernel_size − 1 i n p u t ( N i , C j , s t r i d e × k + m ) out(N_i, C_j, k) = max_{m=0, ldots, text{kernel_size} - 1}
 input(N_i, C_j, stride times k + m)

o u t ( N i ​ , C j ​ , k ) = m = 0 , … , kernel_size − 1 max ​ in p u t ( N i ​ , C j ​ , s t r i d e × k + m )

If `padding`  is non-zero, then the input is implicitly padded with negative infinity on both sides
for `padding`  number of points. `dilation`  is the stride between the elements within the
sliding window. This [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)  has a nice visualization of the pooling parameters. 

Note 

When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
or the input. Sliding windows that would start in the right padded region are ignored.

Parameters
:   * **kernel_size** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – The size of the sliding window, must be > 0.
* **stride** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – The stride of the sliding window, must be > 0. Default value is `kernel_size`  .
* **padding** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
* **dilation** ( [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]*  ) – The stride between elements within a sliding window, must be > 0.
* **return_indices** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If `True`  , will return the argmax along with the max values.
Useful for [`torch.nn.MaxUnpool1d`](torch.nn.MaxUnpool1d.html#torch.nn.MaxUnpool1d "torch.nn.MaxUnpool1d")  later
* **ceil_mode** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If `True`  , will use *ceil* instead of *floor* to compute the output shape. This
ensures that every element in the input tensor is covered by a sliding window.

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
</math> -->( C , L o u t ) (C, L_{out})( C , L o u t ​ )  ,

    where `ceil_mode = False`

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
                       dilation
                      </mtext>
    <mo>
                       ×
                      </mo>
    <mo stretchy="false">
                       (
                      </mo>
    <mtext>
                       kernel_size
                      </mtext>
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
                       −
                      </mo>
    <mn>
                       1
                      </mn>
    </mrow>
    <mtext>
                      stride
                     </mtext>
    </mfrac>
    <mo fence="true">
                     ⌋
                    </mo>
    </mrow>
    <mo>
                    +
                   </mo>
    <mn>
                    1
                   </mn>
    </mrow>
    <annotation encoding="application/x-tex">
                   L_{out} = leftlfloor frac{L_{in} + 2 times text{padding} - text{dilation}
         times (text{kernel_size} - 1) - 1}{text{stride}}rightrfloor + 1
                  </annotation>
    </semantics>
    </math> -->
    L o u t = ⌊ L i n + 2 × padding − dilation × ( kernel_size − 1 ) − 1 stride ⌋ + 1 L_{out} = leftlfloor frac{L_{in} + 2 times text{padding} - text{dilation}
     times (text{kernel_size} - 1) - 1}{text{stride}}rightrfloor + 1

    L o u t ​ = ⌊ stride L in ​ + 2 × padding − dilation × ( kernel_size − 1 ) − 1 ​ ⌋ + 1

    where `ceil_mode = True`

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
                         ⌈
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
                           dilation
                          </mtext>
        <mo>
                           ×
                          </mo>
        <mo stretchy="false">
                           (
                          </mo>
        <mtext>
                           kernel_size
                          </mtext>
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
                           −
                          </mo>
        <mn>
                           1
                          </mn>
        <mo>
                           +
                          </mo>
        <mo stretchy="false">
                           (
                          </mo>
        <mi>
                           s
                          </mi>
        <mi>
                           t
                          </mi>
        <mi>
                           r
                          </mi>
        <mi>
                           i
                          </mi>
        <mi>
                           d
                          </mi>
        <mi>
                           e
                          </mi>
        <mo>
                           −
                          </mo>
        <mn>
                           1
                          </mn>
        <mo stretchy="false">
                           )
                          </mo>
        </mrow>
        <mtext>
                          stride
                         </mtext>
        </mfrac>
        <mo fence="true">
                         ⌉
                        </mo>
        </mrow>
        <mo>
                        +
                       </mo>
        <mn>
                        1
                       </mn>
        </mrow>
        <annotation encoding="application/x-tex">
                       L_{out} = leftlceil frac{L_{in} + 2 times text{padding} - text{dilation}
              times (text{kernel_size} - 1) - 1 + (stride - 1)}{text{stride}}rightrceil + 1
                      </annotation>
        </semantics>
        </math> -->
        L o u t = ⌈ L i n + 2 × padding − dilation × ( kernel_size − 1 ) − 1 + ( s t r i d e − 1 ) stride ⌉ + 1 L_{out} = leftlceil frac{L_{in} + 2 times text{padding} - text{dilation}
         times (text{kernel_size} - 1) - 1 + (stride - 1)}{text{stride}}rightrceil + 1

    L o u t ​ = ⌈ stride L in ​ + 2 × padding − dilation × ( kernel_size − 1 ) − 1 + ( s t r i d e − 1 ) ​ ⌉ + 1

* Ensure that the last pooling starts inside the image, make <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</mrow>
<annotation encoding="application/x-tex">
               L_{out} = L_{out} - 1
              </annotation>
</semantics>
</math> -->L o u t = L o u t − 1 L_{out} = L_{out} - 1L o u t ​ = L o u t ​ − 1  when <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                ∗
               </mo>
<mtext>
                stride
               </mtext>
<mo>
                &gt;
               </mo>
<mo>
                =
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
               (L_{out} - 1) * text{stride} &gt;= L_{in} + text{padding}
              </annotation>
</semantics>
</math> -->( L o u t − 1 ) ∗ stride > = L i n + padding (L_{out} - 1) * text{stride} >= L_{in} + text{padding}( L o u t ​ − 1 ) ∗ stride >= L in ​ + padding  .

Examples: 

```
>>> # pool of size=3, stride=2
>>> m = nn.MaxPool1d(3, stride=2)
>>> input = torch.randn(20, 16, 50)
>>> output = m(input)

```

