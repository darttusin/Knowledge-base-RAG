MultiMarginLoss 
==================================================================

*class* torch.nn. MultiMarginLoss ( *p = 1*  , *margin = 1.0*  , *weight = None*  , *size_average = None*  , *reduce = None*  , *reduction = 'mean'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py#L1513) 
:   Creates a criterion that optimizes a multi-class classification hinge
loss (margin-based loss) between input <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->x xx  (a 2D mini-batch *Tensor* ) and
output <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->y yy  (which is a 1D tensor of target class indices, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<annotation encoding="application/x-tex">
           0 leq y leq text{x.size}(1)-1
          </annotation>
</semantics>
</math> -->0 ≤ y ≤ x.size ( 1 ) − 1 0 leq y leq text{x.size}(1)-10 ≤ y ≤ x.size ( 1 ) − 1  ): 

For each mini-batch sample, the loss in terms of the 1D input <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->x xx  and scalar
output <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->y yy  is: 

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
<mfrac>
<mrow>
<munder>
<mo>
               ∑
              </mo>
<mi>
               i
              </mi>
</munder>
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
<mtext>
              margin
             </mtext>
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
              y
             </mi>
<mo stretchy="false">
              ]
             </mo>
<mo>
              +
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
<msup>
<mo stretchy="false">
               )
              </mo>
<mi>
               p
              </mi>
</msup>
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
           text{loss}(x, y) = frac{sum_i max(0, text{margin} - x[y] + x[i])^p}{text{x.size}(0)}
          </annotation>
</semantics>
</math> -->
loss ( x , y ) = ∑ i max ⁡ ( 0 , margin − x [ y ] + x [ i ] ) p x.size ( 0 ) text{loss}(x, y) = frac{sum_i max(0, text{margin} - x[y] + x[i])^p}{text{x.size}(0)}

loss ( x , y ) = x.size ( 0 ) ∑ i ​ max ( 0 , margin − x [ y ] + x [ i ] ) p ​

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            i
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
           i in left{0, ; cdots , ; text{x.size}(0) - 1right}
          </annotation>
</semantics>
</math> -->i ∈ { 0 , ⋯ , x.size ( 0 ) − 1 } i in left{0, ; cdots , ; text{x.size}(0) - 1right}i ∈ { 0 , ⋯ , x.size ( 0 ) − 1 }  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</mrow>
<annotation encoding="application/x-tex">
           i neq y
          </annotation>
</semantics>
</math> -->i ≠ y i neq yi  = y  . 

Optionally, you can give non-equal weighting on the classes by passing
a 1D `weight`  tensor into the constructor. 

The loss function then becomes: 

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
<mfrac>
<mrow>
<munder>
<mo>
               ∑
              </mo>
<mi>
               i
              </mi>
</munder>
<mi>
              w
             </mi>
<mo stretchy="false">
              [
             </mo>
<mi>
              y
             </mi>
<mo stretchy="false">
              ]
             </mo>
<mo>
              ∗
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
<mtext>
              margin
             </mtext>
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
              y
             </mi>
<mo stretchy="false">
              ]
             </mo>
<mo>
              +
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
<msup>
<mo stretchy="false">
               )
              </mo>
<mi>
               p
              </mi>
</msup>
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
           text{loss}(x, y) = frac{sum_i w[y] * max(0, text{margin} - x[y] + x[i])^p}{text{x.size}(0)}
          </annotation>
</semantics>
</math> -->
loss ( x , y ) = ∑ i w [ y ] ∗ max ⁡ ( 0 , margin − x [ y ] + x [ i ] ) p x.size ( 0 ) text{loss}(x, y) = frac{sum_i w[y] * max(0, text{margin} - x[y] + x[i])^p}{text{x.size}(0)}

loss ( x , y ) = x.size ( 0 ) ∑ i ​ w [ y ] ∗ max ( 0 , margin − x [ y ] + x [ i ] ) p ​

Parameters
:   * **p** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – Has a default value of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->1 11  . <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->1 11  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
                2
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               2
              </annotation>
</semantics>
</math> -->2 22  are the only supported values.

* **margin** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – Has a default value of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->1 11  .

* **weight** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – a manual rescaling weight given to each
class. If given, it has to be a Tensor of size *C* . Otherwise, it is
treated as if having all ones.
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
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C)
              </annotation>
</semantics>
</math> -->( N , C ) (N, C)( N , C )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( C ) (C)( C )  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                N
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               N
              </annotation>
</semantics>
</math> -->N NN  is the batch size and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                C
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               C
              </annotation>
</semantics>
</math> -->C CC  is the number of classes.

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
</math> -->( ) ()( )  , where each value is <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
                0
               </mn>
<mo>
                ≤
               </mo>
<mtext>
                targets
               </mtext>
<mo stretchy="false">
                [
               </mo>
<mi>
                i
               </mi>
<mo stretchy="false">
                ]
               </mo>
<mo>
                ≤
               </mo>
<mi>
                C
               </mi>
<mo>
                −
               </mo>
<mn>
                1
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               0 leq text{targets}[i] leq C-1
              </annotation>
</semantics>
</math> -->0 ≤ targets [ i ] ≤ C − 1 0 leq text{targets}[i] leq C-10 ≤ targets [ i ] ≤ C − 1  .

* Output: scalar. If `reduction`  is `'none'`  , then same shape as the target.

Examples 

```
>>> loss = nn.MultiMarginLoss()
>>> x = torch.tensor([[0.1, 0.2, 0.4, 0.8]])
>>> y = torch.tensor([3])
>>> # 0.25 * ((1-(0.8-0.1)) + (1-(0.8-0.2)) + (1-(0.8-0.4)))
>>> loss(x, y)
tensor(0.32...)

```

