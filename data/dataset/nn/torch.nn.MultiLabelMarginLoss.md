MultiLabelMarginLoss 
============================================================================

*class* torch.nn. MultiLabelMarginLoss ( *size_average = None*  , *reduce = None*  , *reduction = 'mean'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py#L906) 
:   Creates a criterion that optimizes a multi-class multi-classification
hinge loss (margin-based loss) between input <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            x
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           x
          </annotation>
</semantics>
</math> -->x xx  (a 2D mini-batch *Tensor* )
and output <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->y yy  (which is a 2D *Tensor* of target class indices).
For each sample in the mini-batch: 

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
<munder>
<mo>
             ∑
            </mo>
<mrow>
<mi>
              i
             </mi>
<mi>
              j
             </mi>
</mrow>
</munder>
<mfrac>
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
<mn>
              1
             </mn>
<mo>
              −
             </mo>
<mo stretchy="false">
              (
             </mo>
<mi>
              x
             </mi>
<mo stretchy="false">
              [
             </mo>
<mi>
              y
             </mi>
<mo stretchy="false">
              [
             </mo>
<mi>
              j
             </mi>
<mo stretchy="false">
              ]
             </mo>
<mo stretchy="false">
              ]
             </mo>
<mo>
              −
             </mo>
<mi>
              x
             </mi>
<mo stretchy="false">
              [
             </mo>
<mi>
              i
             </mi>
<mo stretchy="false">
              ]
             </mo>
<mo stretchy="false">
              )
             </mo>
<mo stretchy="false">
              )
             </mo>
</mrow>
<mrow>
<mtext>
              x.size
             </mtext>
<mo stretchy="false">
              (
             </mo>
<mn>
              0
             </mn>
<mo stretchy="false">
              )
             </mo>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           text{loss}(x, y) = sum_{ij}frac{max(0, 1 - (x[y[j]] - x[i]))}{text{x.size}(0)}
          </annotation>
</semantics>
</math> -->
loss ( x , y ) = ∑ i j max ⁡ ( 0 , 1 − ( x [ y [ j ] ] − x [ i ] ) ) x.size ( 0 ) text{loss}(x, y) = sum_{ij}frac{max(0, 1 - (x[y[j]] - x[i]))}{text{x.size}(0)}

loss ( x , y ) = ij ∑ ​ x.size ( 0 ) max ( 0 , 1 − ( x [ y [ j ]] − x [ i ])) ​

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            x
           </mi>
<mo>
            ∈
           </mo>
<mrow>
<mo fence="true">
             {
            </mo>
<mn>
             0
            </mn>
<mo separator="true">
             ,
            </mo>
<mtext>
</mtext>
<mo>
             ⋯
            </mo>
<mtext>
</mtext>
<mo separator="true">
             ,
            </mo>
<mtext>
</mtext>
<mtext>
             x.size
            </mtext>
<mo stretchy="false">
             (
            </mo>
<mn>
             0
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
<mo fence="true">
             }
            </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           x in left{0, ; cdots , ; text{x.size}(0) - 1right}
          </annotation>
</semantics>
</math> -->x ∈ { 0 , ⋯ , x.size ( 0 ) − 1 } x in left{0, ; cdots , ; text{x.size}(0) - 1right}x ∈ { 0 , ⋯ , x.size ( 0 ) − 1 }  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            y
           </mi>
<mo>
            ∈
           </mo>
<mrow>
<mo fence="true">
             {
            </mo>
<mn>
             0
            </mn>
<mo separator="true">
             ,
            </mo>
<mtext>
</mtext>
<mo>
             ⋯
            </mo>
<mtext>
</mtext>
<mo separator="true">
             ,
            </mo>
<mtext>
</mtext>
<mtext>
             y.size
            </mtext>
<mo stretchy="false">
             (
            </mo>
<mn>
             0
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
<mo fence="true">
             }
            </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           y in left{0, ; cdots , ; text{y.size}(0) - 1right}
          </annotation>
</semantics>
</math> -->y ∈ { 0 , ⋯ , y.size ( 0 ) − 1 } y in left{0, ; cdots , ; text{y.size}(0) - 1right}y ∈ { 0 , ⋯ , y.size ( 0 ) − 1 }  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
            0
           </mn>
<mo>
            ≤
           </mo>
<mi>
            y
           </mi>
<mo stretchy="false">
            [
           </mo>
<mi>
            j
           </mi>
<mo stretchy="false">
            ]
           </mo>
<mo>
            ≤
           </mo>
<mtext>
            x.size
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mn>
            0
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
<annotation encoding="application/x-tex">
           0 leq y[j] leq text{x.size}(0)-1
          </annotation>
</semantics>
</math> -->0 ≤ y [ j ] ≤ x.size ( 0 ) − 1 0 leq y[j] leq text{x.size}(0)-10 ≤ y [ j ] ≤ x.size ( 0 ) − 1  , and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            i
           </mi>
<mo mathvariant="normal">
            ≠
           </mo>
<mi>
            y
           </mi>
<mo stretchy="false">
            [
           </mo>
<mi>
            j
           </mi>
<mo stretchy="false">
            ]
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           i neq y[j]
          </annotation>
</semantics>
</math> -->i ≠ y [ j ] i neq y[j]i  = y [ j ]  for all <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mi>
            j
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           j
          </annotation>
</semantics>
</math> -->j jj  . 

<!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->y yy  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            x
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           x
          </annotation>
</semantics>
</math> -->x xx  must have the same size. 

The criterion only considers a contiguous block of non-negative targets that
starts at the front. 

This allows for different samples to have variable amounts of target classes. 

Parameters
:   * **size_average** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ). By default,
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
:   * Input: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                C
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (C)
              </annotation>
</semantics>
</math> -->( C ) (C)( C )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C)
              </annotation>
</semantics>
</math> -->( N , C ) (N, C)( N , C )  where *N* is the batch size and *C* is the number of classes.

* Target: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                C
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (C)
              </annotation>
</semantics>
</math> -->( C ) (C)( C )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C)
              </annotation>
</semantics>
</math> -->( N , C ) (N, C)( N , C )  , label targets padded by -1 ensuring same shape as the input.

* Output: scalar. If `reduction`  is `'none'`  , then <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N ) (N)( N )  .

Examples 

```
>>> loss = nn.MultiLabelMarginLoss()
>>> x = torch.FloatTensor([[0.1, 0.2, 0.4, 0.8]])
>>> # for target y, only consider labels 3 and 0, not after label -1
>>> y = torch.LongTensor([[3, 0, -1, 1]])
>>> # 0.25 * ((1-(0.1-0.2)) + (1-(0.1-0.4)) + (1-(0.8-0.2)) + (1-(0.8-0.4)))
>>> loss(x, y)
tensor(0.85...)

```

