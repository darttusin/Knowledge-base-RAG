torch.nn.functional.poisson_nll_loss 
==============================================================================================================

torch.nn.functional. poisson_nll_loss ( *input*  , *target*  , *log_input = True*  , *full = False*  , *size_average = None*  , *eps = 1e-08*  , *reduce = None*  , *reduction = 'mean'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L3152) 
:   Compute the Poisson negative log likelihood loss. 

See [`PoissonNLLLoss`](torch.nn.PoissonNLLLoss.html#torch.nn.PoissonNLLLoss "torch.nn.PoissonNLLLoss")  for details. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Expectation of underlying Poisson distribution.
* **target** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Random sample <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                t
               </mi>
<mi>
                a
               </mi>
<mi>
                r
               </mi>
<mi>
                g
               </mi>
<mi>
                e
               </mi>
<mi>
                t
               </mi>
<mo>
                ∼
               </mo>
<mtext>
                Poisson
               </mtext>
<mo stretchy="false">
                (
               </mo>
<mi>
                i
               </mi>
<mi>
                n
               </mi>
<mi>
                p
               </mi>
<mi>
                u
               </mi>
<mi>
                t
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               target sim text{Poisson}(input)
              </annotation>
</semantics>
</math> -->t a r g e t ∼ Poisson ( i n p u t ) target sim text{Poisson}(input)t a r g e t ∼ Poisson ( in p u t )  .

* **log_input** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If `True`  the loss is computed as <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
               exp(text{input}) - text{target} * text{input}
              </annotation>
</semantics>
</math> -->exp ⁡ ( input ) − target ∗ input exp(text{input}) - text{target} * text{input}exp ( input ) − target ∗ input  , if `False`  then loss is <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
               text{input} - text{target} * log(text{input}+text{eps})
              </annotation>
</semantics>
</math> -->input − target ∗ log ⁡ ( input + eps ) text{input} - text{target} * log(text{input}+text{eps})input − target ∗ lo g ( input + eps )  . Default: `True`

* **full** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – Whether to compute full loss, i. e. to add the Stirling
approximation term. Default: `False` <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mo>
                ∗
               </mo>
<mi>
                π
               </mi>
<mo>
                ∗
               </mo>
<mtext>
                target
               </mtext>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               text{target} * log(text{target}) - text{target} + 0.5 * log(2 * pi * text{target})
              </annotation>
</semantics>
</math> -->target ∗ log ⁡ ( target ) − target + 0.5 ∗ log ⁡ ( 2 ∗ π ∗ target ) text{target} * log(text{target}) - text{target} + 0.5 * log(2 * pi * text{target})target ∗ lo g ( target ) − target + 0.5 ∗ lo g ( 2 ∗ π ∗ target )  .

* **size_average** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ).
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
</math> -->log ⁡ ( 0 ) log(0)lo g ( 0 )  when `log_input`  = `False`  . Default: 1e-8

* **reduce** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ).
* **reduction** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – Specifies the reduction to apply to the output: `'none'`  | `'mean'`  | `'sum'`  . `'none'`  : no reduction will be applied, `'mean'`  : the sum of the output will be divided by the number of
elements in the output, `'sum'`  : the output will be summed. Note: `size_average`  and `reduce`  are in the process of being deprecated, and in the meantime,
specifying either of those two args will override `reduction`  . Default: `'mean'`

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

