CosineEmbeddingLoss 
==========================================================================

*class* torch.nn. CosineEmbeddingLoss ( *margin = 0.0*  , *size_average = None*  , *reduce = None*  , *reduction = 'mean'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py#L1376) 
:   Creates a criterion that measures the loss given input tensors <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             x
            </mi>
<mn>
             1
            </mn>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           x_1
          </annotation>
</semantics>
</math> -->x 1 x_1x 1 ​  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             x
            </mi>
<mn>
             2
            </mn>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           x_2
          </annotation>
</semantics>
</math> -->x 2 x_2x 2 ​  and a *Tensor* label <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            y
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           y
          </annotation>
</semantics>
</math> -->y yy  with values 1 or -1.
Use ( <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            y
           </mi>
<mo>
            =
           </mo>
<mn>
            1
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           y=1
          </annotation>
</semantics>
</math> -->y = 1 y=1y = 1  ) to maximize the cosine similarity of two inputs, and ( <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            y
           </mi>
<mo>
            =
           </mo>
<mo>
            −
           </mo>
<mn>
            1
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           y=-1
          </annotation>
</semantics>
</math> -->y = − 1 y=-1y = − 1  ) otherwise.
This is typically used for learning nonlinear
embeddings or semi-supervised learning. 

The loss function for each sample is: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            loss
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            y
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
<mn>
                  1
                 </mn>
<mo>
                  −
                 </mo>
<mi>
                  cos
                 </mi>
<mo>
                  ⁡
                 </mo>
<mo stretchy="false">
                  (
                 </mo>
<msub>
<mi>
                   x
                  </mi>
<mn>
                   1
                  </mn>
</msub>
<mo separator="true">
                  ,
                 </mo>
<msub>
<mi>
                   x
                  </mi>
<mn>
                   2
                  </mn>
</msub>
<mo stretchy="false">
                  )
                 </mo>
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
                  y
                 </mi>
<mo>
                  =
                 </mo>
<mn>
                  1
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
                  cos
                 </mi>
<mo>
                  ⁡
                 </mo>
<mo stretchy="false">
                  (
                 </mo>
<msub>
<mi>
                   x
                  </mi>
<mn>
                   1
                  </mn>
</msub>
<mo separator="true">
                  ,
                 </mo>
<msub>
<mi>
                   x
                  </mi>
<mn>
                   2
                  </mn>
</msub>
<mo stretchy="false">
                  )
                 </mo>
<mo>
                  −
                 </mo>
<mtext>
                  margin
                 </mtext>
<mo stretchy="false">
                  )
                 </mo>
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
                  y
                 </mi>
<mo>
                  =
                 </mo>
<mo>
                  −
                 </mo>
<mn>
                  1
                 </mn>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           text{loss}(x, y) =
begin{cases}
1 - cos(x_1, x_2), &amp; text{if } y = 1 
max(0, cos(x_1, x_2) - text{margin}), &amp; text{if } y = -1
end{cases}
          </annotation>
</semantics>
</math> -->
loss ( x , y ) = { 1 − cos ⁡ ( x 1 , x 2 ) , if y = 1 max ⁡ ( 0 , cos ⁡ ( x 1 , x 2 ) − margin ) , if y = − 1 text{loss}(x, y) =
begin{cases}
1 - cos(x_1, x_2), & text{if } y = 1 
max(0, cos(x_1, x_2) - text{margin}), & text{if } y = -1
end{cases}

loss ( x , y ) = { 1 − cos ( x 1 ​ , x 2 ​ ) , max ( 0 , cos ( x 1 ​ , x 2 ​ ) − margin ) , ​ if y = 1 if y = − 1 ​

Parameters
:   * **margin** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – Should be a number from <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
                −
               </mo>
<mn>
                1
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               -1
              </annotation>
</semantics>
</math> -->− 1 -1− 1  to <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
                1
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               1
              </annotation>
</semantics>
</math> -->1 11  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
                0
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               0
              </annotation>
</semantics>
</math> -->0 00  to <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
                0.5
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               0.5
              </annotation>
</semantics>
</math> -->0.5 0.50.5  is suggested. If `margin`  is missing, the
default value is <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
                0
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               0
              </annotation>
</semantics>
</math> -->0 00  .

* **size_average** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ). By default,
the losses are averaged over each loss element in the batch. Note that for
some losses, there are multiple elements per sample. If the field `size_average`  is set to `False`  , the losses are instead summed for each minibatch. Ignored
when `reduce`  is `False`  . Default: `True`
* **reduce** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ). By default, the
losses are averaged or summed over observations for each minibatch depending
on `size_average`  . When `reduce`  is `False`  , returns a loss per
batch element instead and ignores `size_average`  . Default: `True`
* **reduction** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – Specifies the reduction to apply to the output: `'none'`  | `'mean'`  | `'sum'`  . `'none'`  : no reduction will be applied, `'mean'`  : the sum of the output will be divided by the number of
elements in the output, `'sum'`  : the output will be summed. Note: `size_average`  and `reduce`  are in the process of being deprecated, and in the meantime,
specifying either of those two args will override `reduction`  . Default: `'mean'`

Shape:
:   * Input1: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                D
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, D)
              </annotation>
</semantics>
</math> -->( N , D ) (N, D)( N , D )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                D
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (D)
              </annotation>
</semantics>
</math> -->( D ) (D)( D )  , where *N* is the batch size and *D* is the embedding dimension.

* Input2: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                D
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, D)
              </annotation>
</semantics>
</math> -->( N , D ) (N, D)( N , D )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                D
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (D)
              </annotation>
</semantics>
</math> -->( D ) (D)( D )  , same shape as Input1.

* Target: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N)
              </annotation>
</semantics>
</math> -->( N ) (N)( N )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               ()
              </annotation>
</semantics>
</math> -->( ) ()( )  .

* Output: If `reduction`  is `'none'`  , then <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N)
              </annotation>
</semantics>
</math> -->( N ) (N)( N )  , otherwise scalar.

Examples 

```
>>> loss = nn.CosineEmbeddingLoss()
>>> input1 = torch.randn(3, 5, requires_grad=True)
>>> input2 = torch.randn(3, 5, requires_grad=True)
>>> target = torch.ones(3)
>>> output = loss(input1, input2, target)
>>> output.backward()

```

