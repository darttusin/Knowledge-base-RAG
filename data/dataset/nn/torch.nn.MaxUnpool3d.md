MaxUnpool3d 
==========================================================

*class* torch.nn. MaxUnpool3d ( *kernel_size*  , *stride = None*  , *padding = 0* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/pooling.py#L500) 
:   Computes a partial inverse of [`MaxPool3d`](torch.nn.MaxPool3d.html#torch.nn.MaxPool3d "torch.nn.MaxPool3d")  . 

[`MaxPool3d`](torch.nn.MaxPool3d.html#torch.nn.MaxPool3d "torch.nn.MaxPool3d")  is not fully invertible, since the non-maximal values are lost. [`MaxUnpool3d`](#torch.nn.MaxUnpool3d "torch.nn.MaxUnpool3d")  takes in as input the output of [`MaxPool3d`](torch.nn.MaxPool3d.html#torch.nn.MaxPool3d "torch.nn.MaxPool3d")  including the indices of the maximal values and computes a partial inverse
in which all non-maximal values are set to zero. 

Note 

This operation may behave nondeterministically when the input indices has repeat values.
See [pytorch/pytorch#80827](https://github.com/pytorch/pytorch/issues/80827)  and [Reproducibility](../notes/randomness.html)  for more information.

Note 

[`MaxPool3d`](torch.nn.MaxPool3d.html#torch.nn.MaxPool3d "torch.nn.MaxPool3d")  can map several input sizes to the same output
sizes. Hence, the inversion process can get ambiguous.
To accommodate this, you can provide the needed output size
as an additional argument `output_size`  in the forward call.
See the Inputs section below.

Parameters
:   * **kernel_size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  ) – Size of the max pooling window.
* **stride** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  ) – Stride of the max pooling window.
It is set to `kernel_size`  by default.
* **padding** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  ) – Padding that was added to the input

Inputs:
:   * *input* : the input Tensor to invert
* *indices* : the indices given out by [`MaxPool3d`](torch.nn.MaxPool3d.html#torch.nn.MaxPool3d "torch.nn.MaxPool3d")
* *output_size* (optional): the targeted output size

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
                 D
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
               (N, C, D_{in}, H_{in}, W_{in})
              </annotation>
</semantics>
</math> -->( N , C , D i n , H i n , W i n ) (N, C, D_{in}, H_{in}, W_{in})( N , C , D in ​ , H in ​ , W in ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                 D
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
               (C, D_{in}, H_{in}, W_{in})
              </annotation>
</semantics>
</math> -->( C , D i n , H i n , W i n ) (C, D_{in}, H_{in}, W_{in})( C , D in ​ , H in ​ , W in ​ )  .

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
                 D
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
               (N, C, D_{out}, H_{out}, W_{out})
              </annotation>
</semantics>
</math> -->( N , C , D o u t , H o u t , W o u t ) (N, C, D_{out}, H_{out}, W_{out})( N , C , D o u t ​ , H o u t ​ , W o u t ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                 D
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
               (C, D_{out}, H_{out}, W_{out})
              </annotation>
</semantics>
</math> -->( C , D o u t , H o u t , W o u t ) (C, D_{out}, H_{out}, W_{out})( C , D o u t ​ , H o u t ​ , W o u t ​ )  , where

    <!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
    <semantics>
    <mrow>
    <msub>
    <mi>
                     D
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
    <mo stretchy="false">
                    (
                   </mo>
    <msub>
    <mi>
                     D
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
                    stride[0]
                   </mtext>
    <mo>
                    −
                   </mo>
    <mn>
                    2
                   </mn>
    <mo>
                    ×
                   </mo>
    <mtext>
                    padding[0]
                   </mtext>
    <mo>
                    +
                   </mo>
    <mtext>
                    kernel_size[0]
                   </mtext>
    </mrow>
    <annotation encoding="application/x-tex">
                   D_{out} = (D_{in} - 1) times text{stride[0]} - 2 times text{padding[0]} + text{kernel_size[0]}
                  </annotation>
    </semantics>
    </math> -->
    D o u t = ( D i n − 1 ) × stride[0] − 2 × padding[0] + kernel_size[0] D_{out} = (D_{in} - 1) times text{stride[0]} - 2 times text{padding[0]} + text{kernel_size[0]}

    D o u t ​ = ( D in ​ − 1 ) × stride[0] − 2 × padding[0] + kernel_size[0]

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
                        stride[1]
                       </mtext>
        <mo>
                        −
                       </mo>
        <mn>
                        2
                       </mn>
        <mo>
                        ×
                       </mo>
        <mtext>
                        padding[1]
                       </mtext>
        <mo>
                        +
                       </mo>
        <mtext>
                        kernel_size[1]
                       </mtext>
        </mrow>
        <annotation encoding="application/x-tex">
                       H_{out} = (H_{in} - 1) times text{stride[1]} - 2 times text{padding[1]} + text{kernel_size[1]}
                      </annotation>
        </semantics>
        </math> -->
        H o u t = ( H i n − 1 ) × stride[1] − 2 × padding[1] + kernel_size[1] H_{out} = (H_{in} - 1) times text{stride[1]} - 2 times text{padding[1]} + text{kernel_size[1]}

    H o u t ​ = ( H in ​ − 1 ) × stride[1] − 2 × padding[1] + kernel_size[1]

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
        <mo stretchy="false">
                        (
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
                        stride[2]
                       </mtext>
        <mo>
                        −
                       </mo>
        <mn>
                        2
                       </mn>
        <mo>
                        ×
                       </mo>
        <mtext>
                        padding[2]
                       </mtext>
        <mo>
                        +
                       </mo>
        <mtext>
                        kernel_size[2]
                       </mtext>
        </mrow>
        <annotation encoding="application/x-tex">
                       W_{out} = (W_{in} - 1) times text{stride[2]} - 2 times text{padding[2]} + text{kernel_size[2]}
                      </annotation>
        </semantics>
        </math> -->
        W o u t = ( W i n − 1 ) × stride[2] − 2 × padding[2] + kernel_size[2] W_{out} = (W_{in} - 1) times text{stride[2]} - 2 times text{padding[2]} + text{kernel_size[2]}

    W o u t ​ = ( W in ​ − 1 ) × stride[2] − 2 × padding[2] + kernel_size[2]

    or as given by `output_size`  in the call operator

Example: 

```
>>> # pool of square window of size=3, stride=2
>>> pool = nn.MaxPool3d(3, stride=2, return_indices=True)
>>> unpool = nn.MaxUnpool3d(3, stride=2)
>>> output, indices = pool(torch.randn(20, 16, 51, 33, 15))
>>> unpooled_output = unpool(output, indices)
>>> unpooled_output.size()
torch.Size([20, 16, 51, 33, 15])

```

