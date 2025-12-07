PoissonNLLLoss 
================================================================

*class* torch.nn. PoissonNLLLoss ( *log_input = True*  , *full = False*  , *size_average = None*  , *eps = 1e-08*  , *reduce = None*  , *reduction = 'mean'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py#L280) 
:   Negative log likelihood loss with Poisson distribution of target. 

The loss can be described as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            target
           </mtext>
<mo>
            ∼
           </mo>
<mrow>
<mi mathvariant="normal">
             P
            </mi>
<mi mathvariant="normal">
             o
            </mi>
<mi mathvariant="normal">
             i
            </mi>
<mi mathvariant="normal">
             s
            </mi>
<mi mathvariant="normal">
             s
            </mi>
<mi mathvariant="normal">
             o
            </mi>
<mi mathvariant="normal">
             n
            </mi>
</mrow>
<mo stretchy="false">
            (
           </mo>
<mtext>
            input
           </mtext>
<mo stretchy="false">
            )
           </mo>
<mtext>
            loss
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mtext>
            input
           </mtext>
<mo separator="true">
            ,
           </mo>
<mtext>
            target
           </mtext>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mtext>
            input
           </mtext>
<mo>
            −
           </mo>
<mtext>
            target
           </mtext>
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
<mtext>
            input
           </mtext>
<mo stretchy="false">
            )
           </mo>
<mo>
            +
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
<mtext>
            target!
           </mtext>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{target} sim mathrm{Poisson}(text{input})

text{loss}(text{input}, text{target}) = text{input} - text{target} * log(text{input})
                            + log(text{target!})
          </annotation>
</semantics>
</math> -->
target ∼ P o i s s o n ( input ) loss ( input , target ) = input − target ∗ log ⁡ ( input ) + log ⁡ ( target! ) text{target} sim mathrm{Poisson}(text{input})

text{loss}(text{input}, text{target}) = text{input} - text{target} * log(text{input})
 + log(text{target!})

target ∼ Poisson ( input ) loss ( input , target ) = input − target ∗ lo g ( input ) + lo g ( target! )

The last term can be omitted or approximated with Stirling formula. The
approximation is used for target values more than 1. For targets less or
equal to 1 zeros are added to the loss. 

Parameters
:   * **log_input** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – if `True`  the loss is computed as <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
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
<mtext>
                input
               </mtext>
<mo stretchy="false">
                )
               </mo>
<mo>
                −
               </mo>
<mtext>
                target
               </mtext>
<mo>
                ∗
               </mo>
<mtext>
                input
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               exp(text{input}) - text{target}*text{input}
              </annotation>
</semantics>
</math> -->exp ⁡ ( input ) − target ∗ input exp(text{input}) - text{target}*text{input}exp ( input ) − target ∗ input  , if `False`  the loss is <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
                input
               </mtext>
<mo>
                −
               </mo>
<mtext>
                target
               </mtext>
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
<mtext>
                input
               </mtext>
<mo>
                +
               </mo>
<mtext>
                eps
               </mtext>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               text{input} - text{target}*log(text{input}+text{eps})
              </annotation>
</semantics>
</math> -->input − target ∗ log ⁡ ( input + eps ) text{input} - text{target}*log(text{input}+text{eps})input − target ∗ lo g ( input + eps )  .

* **full** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) –

    whether to compute full loss, i. e. to add the
        Stirling approximation term

    <!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
    <semantics>
    <mrow>
    <mtext>
                    target
                   </mtext>
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
    <mtext>
                    target
                   </mtext>
    <mo stretchy="false">
                    )
                   </mo>
    <mo>
                    −
                   </mo>
    <mtext>
                    target
                   </mtext>
    <mo>
                    +
                   </mo>
    <mn>
                    0.5
                   </mn>
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
    <mn>
                    2
                   </mn>
    <mi>
                    π
                   </mi>
    <mtext>
                    target
                   </mtext>
    <mo stretchy="false">
                    )
                   </mo>
    <mi mathvariant="normal">
                    .
                   </mi>
    </mrow>
    <annotation encoding="application/x-tex">
                   text{target}*log(text{target}) - text{target} + 0.5 * log(2pitext{target}).
                  </annotation>
    </semantics>
    </math> -->
    target ∗ log ⁡ ( target ) − target + 0.5 ∗ log ⁡ ( 2 π target ) . text{target}*log(text{target}) - text{target} + 0.5 * log(2pitext{target}).

    target ∗ lo g ( target ) − target + 0.5 ∗ lo g ( 2 π target ) .

* **size_average** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ). By default,
the losses are averaged over each loss element in the batch. Note that for
some losses, there are multiple elements per sample. If the field `size_average`  is set to `False`  , the losses are instead summed for each minibatch. Ignored
when `reduce`  is `False`  . Default: `True`
* **eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – Small value to avoid evaluation of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
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
                0
               </mn>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               log(0)
              </annotation>
</semantics>
</math> -->log ⁡ ( 0 ) log(0)lo g ( 0 )  when `log_input = False`  . Default: 1e-8

* **reduce** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ). By default, the
losses are averaged or summed over observations for each minibatch depending
on `size_average`  . When `reduce`  is `False`  , returns a loss per
batch element instead and ignores `size_average`  . Default: `True`
* **reduction** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – Specifies the reduction to apply to the output: `'none'`  | `'mean'`  | `'sum'`  . `'none'`  : no reduction will be applied, `'mean'`  : the sum of the output will be divided by the number of
elements in the output, `'sum'`  : the output will be summed. Note: `size_average`  and `reduce`  are in the process of being deprecated, and in the meantime,
specifying either of those two args will override `reduction`  . Default: `'mean'`

Examples 

```
>>> loss = nn.PoissonNLLLoss()
>>> log_input = torch.randn(5, 2, requires_grad=True)
>>> target = torch.randn(5, 2)
>>> output = loss(log_input, target)
>>> output.backward()

```

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

* Output: scalar by default. If `reduction`  is `'none'`  , then <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( ∗ ) (*)( ∗ )  ,
the same shape as the input.

