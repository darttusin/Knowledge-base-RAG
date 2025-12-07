SoftMarginLoss 
================================================================

*class* torch.nn. SoftMarginLoss ( *size_average = None*  , *reduce = None*  , *reduction = 'mean'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py#L1116) 
:   Creates a criterion that optimizes a two-class classification
logistic loss between input tensor <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->x xx  and target tensor <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->y yy  (containing 1 or -1). 

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
<mi>
             i
            </mi>
</munder>
<mfrac>
<mrow>
<mi>
              log
             </mi>
<mo>
              ⁡
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
              x.nelement
             </mtext>
<mo stretchy="false">
              (
             </mo>
<mo stretchy="false">
              )
             </mo>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           text{loss}(x, y) = sum_i frac{log(1 + exp(-y[i]*x[i]))}{text{x.nelement}()}
          </annotation>
</semantics>
</math> -->
loss ( x , y ) = ∑ i log ⁡ ( 1 + exp ⁡ ( − y [ i ] ∗ x [ i ] ) ) x.nelement ( ) text{loss}(x, y) = sum_i frac{log(1 + exp(-y[i]*x[i]))}{text{x.nelement}()}

loss ( x , y ) = i ∑ ​ x.nelement ( ) lo g ( 1 + exp ( − y [ i ] ∗ x [ i ])) ​

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
</math> -->( ∗ ) (*)( ∗ )  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->∗ *∗  means any number of dimensions.

* Target: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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

* Output: scalar. If `reduction`  is `'none'`  , then <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( ∗ ) (*)( ∗ )  , same
shape as input.

