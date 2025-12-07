torch.nn.functional.pad 
==================================================================================

torch.nn.functional. pad ( *input*  , *pad*  , *mode = 'constant'*  , *value = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L5209) 
:   Pads tensor. 

Padding size:
:   The padding size by which to pad some dimensions of `input`  are described starting from the last dimension and moving forward. <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo fence="true">
              ⌊
             </mo>
<mfrac>
<mtext>
               len(pad)
              </mtext>
<mn>
               2
              </mn>
</mfrac>
<mo fence="true">
              ⌋
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             leftlfloorfrac{text{len(pad)}}{2}rightrfloor
            </annotation>
</semantics>
</math> -->⌊ len(pad) 2 ⌋ leftlfloorfrac{text{len(pad)}}{2}rightrfloor⌊ 2 len(pad) ​ ⌋  dimensions
of `input`  will be padded.
For example, to pad only the last dimension of the input tensor, then [`pad`](#torch.nn.functional.pad "torch.nn.functional.pad")  has the form <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mtext>
              padding_left
             </mtext>
<mo separator="true">
              ,
             </mo>
<mtext>
              padding_right
             </mtext>
<mo stretchy="false">
              )
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             (text{padding_left}, text{padding_right})
            </annotation>
</semantics>
</math> -->( padding_left , padding_right ) (text{padding_left}, text{padding_right})( padding_left , padding_right )  ;
to pad the last 2 dimensions of the input tensor, then use <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mtext>
              padding_left
             </mtext>
<mo separator="true">
              ,
             </mo>
<mtext>
              padding_right
             </mtext>
<mo separator="true">
              ,
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             (text{padding_left}, text{padding_right},
            </annotation>
</semantics>
</math> -->( padding_left , padding_right , (text{padding_left}, text{padding_right},( padding_left , padding_right , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
              padding_top
             </mtext>
<mo separator="true">
              ,
             </mo>
<mtext>
              padding_bottom
             </mtext>
<mo stretchy="false">
              )
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             text{padding_top}, text{padding_bottom})
            </annotation>
</semantics>
</math> -->padding_top , padding_bottom ) text{padding_top}, text{padding_bottom})padding_top , padding_bottom )  ;
to pad the last 3 dimensions, use <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mtext>
              padding_left
             </mtext>
<mo separator="true">
              ,
             </mo>
<mtext>
              padding_right
             </mtext>
<mo separator="true">
              ,
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             (text{padding_left}, text{padding_right},
            </annotation>
</semantics>
</math> -->( padding_left , padding_right , (text{padding_left}, text{padding_right},( padding_left , padding_right , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
              padding_top
             </mtext>
<mo separator="true">
              ,
             </mo>
<mtext>
              padding_bottom
             </mtext>
</mrow>
<annotation encoding="application/x-tex">
             text{padding_top}, text{padding_bottom}
            </annotation>
</semantics>
</math> -->padding_top , padding_bottom text{padding_top}, text{padding_bottom}padding_top , padding_bottom <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
              padding_front
             </mtext>
<mo separator="true">
              ,
             </mo>
<mtext>
              padding_back
             </mtext>
<mo stretchy="false">
              )
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             text{padding_front}, text{padding_back})
            </annotation>
</semantics>
</math> -->padding_front , padding_back ) text{padding_front}, text{padding_back})padding_front , padding_back )  .

Padding mode:
:   See [`torch.nn.CircularPad2d`](torch.nn.CircularPad2d.html#torch.nn.CircularPad2d "torch.nn.CircularPad2d")  , [`torch.nn.ConstantPad2d`](torch.nn.ConstantPad2d.html#torch.nn.ConstantPad2d "torch.nn.ConstantPad2d")  , [`torch.nn.ReflectionPad2d`](torch.nn.ReflectionPad2d.html#torch.nn.ReflectionPad2d "torch.nn.ReflectionPad2d")  , and [`torch.nn.ReplicationPad2d`](torch.nn.ReplicationPad2d.html#torch.nn.ReplicationPad2d "torch.nn.ReplicationPad2d")  for concrete examples on how each of the padding modes works. Constant
padding is implemented for arbitrary dimensions. Circular, replicate and
reflection padding are implemented for padding the last 3 dimensions of a
4D or 5D input tensor, the last 2 dimensions of a 3D or 4D input tensor,
or the last dimension of a 2D or 3D input tensor.

Note 

When using the CUDA backend, this operation may induce nondeterministic
behaviour in its backward pass that is not easily switched off.
Please see the notes on [Reproducibility](../notes/randomness.html)  for background.

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – N-dimensional tensor
* **pad** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  ) – m-elements tuple, where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mfrac>
<mi>
                 m
                </mi>
<mn>
                 2
                </mn>
</mfrac>
<mo>
                ≤
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               frac{m}{2} leq
              </annotation>
</semantics>
</math> -->m 2 ≤ frac{m}{2} leq2 m ​ ≤  input dimensions and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                m
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               m
              </annotation>
</semantics>
</math> -->m mm  is even.

* **mode** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – `'constant'`  , `'reflect'`  , `'replicate'`  or `'circular'`  .
Default: `'constant'`
* **value** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *]*  ) – fill value for `'constant'`  padding. Default: `0`

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

Examples: 

```
>>> t4d = torch.empty(3, 3, 4, 2)
>>> p1d = (1, 1) # pad last dim by 1 on each side
>>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
>>> print(out.size())
torch.Size([3, 3, 4, 4])
>>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
>>> out = F.pad(t4d, p2d, "constant", 0)
>>> print(out.size())
torch.Size([3, 3, 8, 4])
>>> t4d = torch.empty(3, 3, 4, 2)
>>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
>>> out = F.pad(t4d, p3d, "constant", 0)
>>> print(out.size())
torch.Size([3, 9, 7, 3])

```

