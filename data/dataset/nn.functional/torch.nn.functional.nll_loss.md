torch.nn.functional.nll_loss 
=============================================================================================

torch.nn.functional. nll_loss ( *input*  , *target*  , *weight = None*  , *size_average = None*  , *ignore_index = -100*  , *reduce = None*  , *reduction = 'mean'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L3090) 
:   Compute the negative log likelihood loss. 

See [`NLLLoss`](torch.nn.NLLLoss.html#torch.nn.NLLLoss "torch.nn.NLLLoss")  for details. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N , C ) (N, C)( N , C )  where *C = number of classes* or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mo separator="true">
                ,
               </mo>
<mi>
                H
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                W
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, H, W)
              </annotation>
</semantics>
</math> -->( N , C , H , W ) (N, C, H, W)( N , C , H , W )  in case of 2D Loss, or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 d
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
                 d
                </mi>
<mn>
                 2
                </mn>
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
                 d
                </mi>
<mi>
                 K
                </mi>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, d_1, d_2, ..., d_K)
              </annotation>
</semantics>
</math> -->( N , C , d 1 , d 2 , . . . , d K ) (N, C, d_1, d_2, ..., d_K)( N , C , d 1 ​ , d 2 ​ , ... , d K ​ )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                K
               </mi>
<mo>
                ≥
               </mo>
<mn>
                1
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               K geq 1
              </annotation>
</semantics>
</math> -->K ≥ 1 K geq 1K ≥ 1  in the case of K-dimensional loss. *input* is expected to be log-probabilities.

* **target** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N ) (N)( N )  where each value is <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->0 ≤ targets [ i ] ≤ C − 1 0 leq text{targets}[i] leq C-10 ≤ targets [ i ] ≤ C − 1  ,
or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<msub>
<mi>
                 d
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
                 d
                </mi>
<mn>
                 2
                </mn>
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
                 d
                </mi>
<mi>
                 K
                </mi>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, d_1, d_2, ..., d_K)
              </annotation>
</semantics>
</math> -->( N , d 1 , d 2 , . . . , d K ) (N, d_1, d_2, ..., d_K)( N , d 1 ​ , d 2 ​ , ... , d K ​ )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                K
               </mi>
<mo>
                ≥
               </mo>
<mn>
                1
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               K geq 1
              </annotation>
</semantics>
</math> -->K ≥ 1 K geq 1K ≥ 1  for
K-dimensional loss.

* **weight** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – A manual rescaling weight given to each
class. If given, has to be a Tensor of size *C*
* **size_average** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ).
* **ignore_index** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – Specifies a target value that is ignored
and does not contribute to the input gradient. When `size_average`  is `True`  , the loss is averaged over non-ignored targets. Default: -100
* **reduce** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ).
* **reduction** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – Specifies the reduction to apply to the output: `'none'`  | `'mean'`  | `'sum'`  . `'none'`  : no reduction will be applied, `'mean'`  : the sum of the output will be divided by the number of
elements in the output, `'sum'`  : the output will be summed. Note: `size_average`  and `reduce`  are in the process of being deprecated, and in the meantime,
specifying either of those two args will override `reduction`  . Default: `'mean'`

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

Example: 

```
>>> # input is of size N x C = 3 x 5
>>> input = torch.randn(3, 5, requires_grad=True)
>>> # each element in target has to have 0 <= value < C
>>> target = torch.tensor([1, 0, 4])
>>> output = F.nll_loss(F.log_softmax(input, dim=1), target)
>>> output.backward()

```

