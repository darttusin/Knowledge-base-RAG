Flatten 
==================================================

*class* torch.nn. Flatten ( *start_dim = 1*  , *end_dim = -1* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/flatten.py#L13) 
:   Flattens a contiguous range of dims into a tensor. 

For use with `Sequential`  , see [`torch.flatten()`](torch.flatten.html#torch.flatten "torch.flatten")  for details. 

Shape:
:   * Input: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 S
                </mi>
<mtext>
                 start
                </mtext>
</msub>
<mo separator="true">
                ,
               </mo>
<mi mathvariant="normal">
                .
               </mi>
<mi mathvariant="normal">
                .
               </mi>
<mi mathvariant="normal">
                .
               </mi>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 S
                </mi>
<mi>
                 i
                </mi>
</msub>
<mo separator="true">
                ,
               </mo>
<mi mathvariant="normal">
                .
               </mi>
<mi mathvariant="normal">
                .
               </mi>
<mi mathvariant="normal">
                .
               </mi>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 S
                </mi>
<mtext>
                 end
                </mtext>
</msub>
<mo separator="true">
                ,
               </mo>
<mo>
                ∗
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (*, S_{text{start}},..., S_{i}, ..., S_{text{end}}, *)
              </annotation>
</semantics>
</math> -->( ∗ , S start , . . . , S i , . . . , S end , ∗ ) (*, S_{text{start}},..., S_{i}, ..., S_{text{end}}, *)( ∗ , S start ​ , ... , S i ​ , ... , S end ​ , ∗ )  ,’
where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
                 S
                </mi>
<mi>
                 i
                </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
               S_{i}
              </annotation>
</semantics>
</math> -->S i S_{i}S i ​  is the size at dimension <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                i
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               i
              </annotation>
</semantics>
</math> -->i ii  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
                ∗
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               *
              </annotation>
</semantics>
</math> -->∗ *∗  means any
number of dimensions including none.

* Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo separator="true">
                ,
               </mo>
<msubsup>
<mo>
                 ∏
                </mo>
<mrow>
<mi>
                  i
                 </mi>
<mo>
                  =
                 </mo>
<mtext>
                  start
                 </mtext>
</mrow>
<mtext>
                 end
                </mtext>
</msubsup>
<msub>
<mi>
                 S
                </mi>
<mi>
                 i
                </mi>
</msub>
<mo separator="true">
                ,
               </mo>
<mo>
                ∗
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (*, prod_{i=text{start}}^{text{end}} S_{i}, *)
              </annotation>
</semantics>
</math> -->( ∗ , ∏ i = start end S i , ∗ ) (*, prod_{i=text{start}}^{text{end}} S_{i}, *)( ∗ , ∏ i = start end ​ S i ​ , ∗ )  .

Parameters
:   * **start_dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – first dim to flatten (default = 1).
* **end_dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – last dim to flatten (default = -1).

Examples::
:   ```
>>> input = torch.randn(32, 1, 5, 5)
>>> # With default parameters
>>> m = nn.Flatten()
>>> output = m(input)
>>> output.size()
torch.Size([32, 25])
>>> # With non-default parameters
>>> m = nn.Flatten(0, 2)
>>> output = m(input)
>>> output.size()
torch.Size([160, 5])

```

