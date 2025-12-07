PReLU 
==============================================

*class* torch.nn. PReLU ( *num_parameters = 1*  , *init = 0.25*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L1457) 
:   Applies the element-wise PReLU function. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            PReLU
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mi>
            max
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<mn>
            0
           </mn>
<mo separator="true">
            ,
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            +
           </mo>
<mi>
            a
           </mi>
<mo>
            ∗
           </mo>
<mi>
            min
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<mn>
            0
           </mn>
<mo separator="true">
            ,
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{PReLU}(x) = max(0,x) + a * min(0,x)
          </annotation>
</semantics>
</math> -->
PReLU ( x ) = max ⁡ ( 0 , x ) + a ∗ min ⁡ ( 0 , x ) text{PReLU}(x) = max(0,x) + a * min(0,x)

PReLU ( x ) = max ( 0 , x ) + a ∗ min ( 0 , x )

or 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            PReLU
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mrow>
<mo fence="true">
             {
            </mo>
<mtable columnalign="left left" columnspacing="1em" rowspacing="0.36em">
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mi>
                  x
                 </mi>
<mo separator="true">
                  ,
                 </mo>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mtext>
                  if
                 </mtext>
<mi>
                  x
                 </mi>
<mo>
                  ≥
                 </mo>
<mn>
                  0
                 </mn>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mi>
                  a
                 </mi>
<mi>
                  x
                 </mi>
<mo separator="true">
                  ,
                 </mo>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mtext>
                 otherwise
                </mtext>
</mstyle>
</mtd>
</mtr>
</mtable>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           text{PReLU}(x) =
begin{cases}
x, &amp; text{ if } x ge 0 
ax, &amp; text{ otherwise }
end{cases}
          </annotation>
</semantics>
</math> -->
PReLU ( x ) = { x , if x ≥ 0 a x , otherwise text{PReLU}(x) =
begin{cases}
x, & text{ if } x ge 0 
ax, & text{ otherwise }
end{cases}

PReLU ( x ) = { x , a x , ​ if x ≥ 0 otherwise ​

Here <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            a
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           a
          </annotation>
</semantics>
</math> -->a aa  is a learnable parameter. When called without arguments, *nn.PReLU()* uses a single
parameter <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            a
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           a
          </annotation>
</semantics>
</math> -->a aa  across all input channels. If called with *nn.PReLU(nChannels)* ,
a separate <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            a
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           a
          </annotation>
</semantics>
</math> -->a aa  is used for each input channel. 

Note 

weight decay should not be used when learning <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             a
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            a
           </annotation>
</semantics>
</math> -->a aa  for good performance.

Note 

Channel dim is the 2nd dim of input. When input has dims < 2, then there is
no channel dim and the number of channels = 1.

Parameters
:   * **num_parameters** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – number of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                a
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               a
              </annotation>
</semantics>
</math> -->a aa  to learn.
Although it takes an int as input, there is only two values are legitimate:
1, or the number of channels at input. Default: 1

* **init** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the initial value of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                a
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               a
              </annotation>
</semantics>
</math> -->a aa  . Default: 0.25

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
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               ( *)
              </annotation>
</semantics>
</math> -->( ∗ ) ( *)( ∗ )  where *** means, any number of additional
dimensions.

* Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (*)
              </annotation>
</semantics>
</math> -->( ∗ ) (*)( ∗ )  , same shape as the input.

Variables
: **weight** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the learnable weights of shape ( `num_parameters`  ).

![../_images/PReLU.png](../_images/PReLU.png)

Examples: 

```
>>> m = nn.PReLU()
>>> input = torch.randn(2)
>>> output = m(input)

```

