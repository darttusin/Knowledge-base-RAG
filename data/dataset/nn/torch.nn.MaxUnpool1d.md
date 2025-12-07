MaxUnpool1d 
==========================================================

*class* torch.nn. MaxUnpool1d ( *kernel_size*  , *stride = None*  , *padding = 0* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/pooling.py#L323) 
:   Computes a partial inverse of [`MaxPool1d`](torch.nn.MaxPool1d.html#torch.nn.MaxPool1d "torch.nn.MaxPool1d")  . 

[`MaxPool1d`](torch.nn.MaxPool1d.html#torch.nn.MaxPool1d "torch.nn.MaxPool1d")  is not fully invertible, since the non-maximal values are lost. 

[`MaxUnpool1d`](#torch.nn.MaxUnpool1d "torch.nn.MaxUnpool1d")  takes in as input the output of [`MaxPool1d`](torch.nn.MaxPool1d.html#torch.nn.MaxPool1d "torch.nn.MaxPool1d")  including the indices of the maximal values and computes a partial inverse
in which all non-maximal values are set to zero. 

Note 

This operation may behave nondeterministically when the input indices has repeat values.
See [pytorch/pytorch#80827](https://github.com/pytorch/pytorch/issues/80827)  and [Reproducibility](../notes/randomness.html)  for more information.

Note 

[`MaxPool1d`](torch.nn.MaxPool1d.html#torch.nn.MaxPool1d "torch.nn.MaxPool1d")  can map several input sizes to the same output
sizes. Hence, the inversion process can get ambiguous.
To accommodate this, you can provide the needed output size
as an additional argument `output_size`  in the forward call.
See the Inputs and Example below.

Parameters
:   * **kernel_size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  ) – Size of the max pooling window.
* **stride** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  ) – Stride of the max pooling window.
It is set to `kernel_size`  by default.
* **padding** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  ) – Padding that was added to the input

Inputs:
:   * *input* : the input Tensor to invert
* *indices* : the indices given out by [`MaxPool1d`](torch.nn.MaxPool1d.html#torch.nn.MaxPool1d "torch.nn.MaxPool1d")
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
</mrow>
<annotation encoding="application/x-tex">
               (N, C, H_{in})
              </annotation>
</semantics>
</math> -->( N , C , H i n ) (N, C, H_{in})( N , C , H in ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (C, H_{in})
              </annotation>
</semantics>
</math> -->( C , H i n ) (C, H_{in})( C , H in ​ )  .

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
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, H_{out})
              </annotation>
</semantics>
</math> -->( N , C , H o u t ) (N, C, H_{out})( N , C , H o u t ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (C, H_{out})
              </annotation>
</semantics>
</math> -->( C , H o u t ) (C, H_{out})( C , H o u t ​ )  , where

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
                    padding
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
    <mo>
                    +
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
    <annotation encoding="application/x-tex">
                   H_{out} = (H_{in} - 1) times text{stride}[0] - 2 times text{padding}[0] + text{kernel_size}[0]
                  </annotation>
    </semantics>
    </math> -->
    H o u t = ( H i n − 1 ) × stride [ 0 ] − 2 × padding [ 0 ] + kernel_size [ 0 ] H_{out} = (H_{in} - 1) times text{stride}[0] - 2 times text{padding}[0] + text{kernel_size}[0]

    H o u t ​ = ( H in ​ − 1 ) × stride [ 0 ] − 2 × padding [ 0 ] + kernel_size [ 0 ]

    or as given by `output_size`  in the call operator

Example: 

```
>>> pool = nn.MaxPool1d(2, stride=2, return_indices=True)
>>> unpool = nn.MaxUnpool1d(2, stride=2)
>>> input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8]]])
>>> output, indices = pool(input)
>>> unpool(output, indices)
tensor([[[ 0.,  2.,  0.,  4.,  0.,  6.,  0., 8.]]])

>>> # Example showcasing the use of output_size
>>> input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8, 9]]])
>>> output, indices = pool(input)
>>> unpool(output, indices, output_size=input.size())
tensor([[[ 0.,  2.,  0.,  4.,  0.,  6.,  0., 8.,  0.]]])

>>> unpool(output, indices)
tensor([[[ 0.,  2.,  0.,  4.,  0.,  6.,  0., 8.]]])

```

