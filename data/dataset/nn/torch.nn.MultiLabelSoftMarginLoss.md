MultiLabelSoftMarginLoss 
====================================================================================

*class* torch.nn. MultiLabelSoftMarginLoss ( *weight = None*  , *size_average = None*  , *reduce = None*  , *reduction = 'mean'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py#L1320) 
:   Creates a criterion that optimizes a multi-label one-versus-all
loss based on max-entropy, between input <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->x xx  and target <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->y yy  of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N , C ) (N, C)( N , C )  .
For each sample in the minibatch: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            l
           </mi>
<mi>
            o
           </mi>
<mi>
            s
           </mi>
<mi>
            s
           </mi>
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
<mo>
            −
           </mo>
<mfrac>
<mn>
             1
            </mn>
<mi>
             C
            </mi>
</mfrac>
<mo>
            ∗
           </mo>
<munder>
<mo>
             ∑
            </mo>
<mi>
             i
            </mi>
</munder>
<mi>
            y
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
<mo>
            ∗
           </mo>
<mi>
            log
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<mo stretchy="false">
            (
           </mo>
<mn>
            1
           </mn>
<mo>
            +
           </mo>
<mi>
            exp
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
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
<msup>
<mo stretchy="false">
             )
            </mo>
<mrow>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
</mrow>
</msup>
<mo stretchy="false">
            )
           </mo>
<mo>
            +
           </mo>
<mo stretchy="false">
            (
           </mo>
<mn>
            1
           </mn>
<mo>
            −
           </mo>
<mi>
            y
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
<mo>
            ∗
           </mo>
<mi>
            log
           </mi>
<mo>
            ⁡
           </mo>
<mrow>
<mo fence="true">
             (
            </mo>
<mfrac>
<mrow>
<mi>
               exp
              </mi>
<mo>
               ⁡
              </mo>
<mo stretchy="false">
               (
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
</mrow>
<mrow>
<mo stretchy="false">
               (
              </mo>
<mn>
               1
              </mn>
<mo>
               +
              </mo>
<mi>
               exp
              </mi>
<mo>
               ⁡
              </mo>
<mo stretchy="false">
               (
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
</mfrac>
<mo fence="true">
             )
            </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           loss(x, y) = - frac{1}{C} * sum_i y[i] * log((1 + exp(-x[i]))^{-1})
                 + (1-y[i]) * logleft(frac{exp(-x[i])}{(1 + exp(-x[i]))}right)
          </annotation>
</semantics>
</math> -->
l o s s ( x , y ) = − 1 C ∗ ∑ i y [ i ] ∗ log ⁡ ( ( 1 + exp ⁡ ( − x [ i ] ) ) − 1 ) + ( 1 − y [ i ] ) ∗ log ⁡ ( exp ⁡ ( − x [ i ] ) ( 1 + exp ⁡ ( − x [ i ] ) ) ) loss(x, y) = - frac{1}{C} * sum_i y[i] * log((1 + exp(-x[i]))^{-1})
 + (1-y[i]) * logleft(frac{exp(-x[i])}{(1 + exp(-x[i]))}right)

l oss ( x , y ) = − C 1 ​ ∗ i ∑ ​ y [ i ] ∗ lo g (( 1 + exp ( − x [ i ]) ) − 1 ) + ( 1 − y [ i ]) ∗ lo g ( ( 1 + exp ( − x [ i ])) exp ( − x [ i ]) ​ )

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
             x.nElement
            </mtext>
<mo stretchy="false">
             (
            </mo>
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
           i in left{0, ; cdots , ; text{x.nElement}() - 1right}
          </annotation>
</semantics>
</math> -->i ∈ { 0 , ⋯ , x.nElement ( ) − 1 } i in left{0, ; cdots , ; text{x.nElement}() - 1right}i ∈ { 0 , ⋯ , x.nElement ( ) − 1 }  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            y
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
<mn>
             1
            </mn>
<mo fence="true">
             }
            </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           y[i] in left{0, ; 1right}
          </annotation>
</semantics>
</math> -->y [ i ] ∈ { 0 , 1 } y[i] in left{0, ; 1right}y [ i ] ∈ { 0 , 1 }  . 

Parameters
:   * **weight** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – a manual rescaling weight given to each
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
</math> -->( N , C ) (N, C)( N , C )  where *N* is the batch size and *C* is the number of classes.

* Target: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N , C ) (N, C)( N , C )  , label targets must have the same shape as the input.

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

