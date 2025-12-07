torch.nn.functional.cross_entropy 
=======================================================================================================

torch.nn.functional. cross_entropy ( *input*  , *target*  , *weight = None*  , *size_average = None*  , *ignore_index = -100*  , *reduce = None*  , *reduction = 'mean'*  , *label_smoothing = 0.0* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L3379) 
:   Compute the cross entropy loss between input logits and target. 

See [`CrossEntropyLoss`](torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss "torch.nn.CrossEntropyLoss")  for details. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Predicted unnormalized logits;
see Shape section below for supported shapes.
* **target** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Ground truth class indices or class probabilities;
see Shape section below for supported shapes.
* **weight** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – a manual rescaling weight given to each
class. If given, has to be a Tensor of size *C*
* **size_average** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ).
* **ignore_index** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – Specifies a target value that is ignored
and does not contribute to the input gradient. When `size_average`  is `True`  , the loss is averaged over non-ignored targets. Note that `ignore_index`  is only applicable when the target contains class indices.
Default: -100
* **reduce** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ).
* **reduction** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – Specifies the reduction to apply to the output: `'none'`  | `'mean'`  | `'sum'`  . `'none'`  : no reduction will be applied, `'mean'`  : the sum of the output will be divided by the number of
elements in the output, `'sum'`  : the output will be summed. Note: `size_average`  and `reduce`  are in the process of being deprecated, and in the meantime,
specifying either of those two args will override `reduction`  . Default: `'mean'`
* **label_smoothing** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – A float in [0.0, 1.0]. Specifies the amount
of smoothing when computing the loss, where 0.0 means no smoothing. The targets
become a mixture of the original ground truth and a uniform distribution as described in [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)  . Default: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
                0.0
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               0.0
              </annotation>
</semantics>
</math> -->0.0 0.00.0  .

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

Shape:
:   * Input: Shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( C ) (C)( C )  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N , C , d 1 , d 2 , . . . , d K ) (N, C, d_1, d_2, ..., d_K)( N , C , d 1 ​ , d 2 ​ , ... , d K ​ )  with <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->K ≥ 1 K geq 1K ≥ 1  in the case of *K* -dimensional loss.

* Target: If containing class indices, shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( ) ()( )  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N , d 1 , d 2 , . . . , d K ) (N, d_1, d_2, ..., d_K)( N , d 1 ​ , d 2 ​ , ... , d K ​ )  with <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->K ≥ 1 K geq 1K ≥ 1  in the case of K-dimensional loss where each value should be between <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                [
               </mo>
<mn>
                0
               </mn>
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
               [0, C)
              </annotation>
</semantics>
</math> -->[ 0 , C ) [0, C)[ 0 , C )  .
If containing class probabilities, same shape as the input and each value should be between <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                [
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
<mo stretchy="false">
                ]
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               [0, 1]
              </annotation>
</semantics>
</math> -->[ 0 , 1 ] [0, 1][ 0 , 1 ]  .

where: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right left" columnspacing="0em" rowspacing="0.25em">
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mi>
                  C
                 </mi>
<mo>
                  =
                 </mo>
<mrow>
</mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mtext>
                  number of classes
                 </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mi>
                  N
                 </mi>
<mo>
                  =
                 </mo>
<mrow>
</mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mtext>
                  batch size
                 </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
             begin{aligned}
    C ={} &amp; text{number of classes} 
    N ={} &amp; text{batch size} 
end{aligned}
            </annotation>
</semantics>
</math> -->
C = number of classes N = batch size begin{aligned}
 C ={} & text{number of classes} 
 N ={} & text{batch size} 
end{aligned}

C = N = ​ number of classes batch size ​

Examples: 

```
>>> # Example of target with class indices
>>> input = torch.randn(3, 5, requires_grad=True)
>>> target = torch.randint(5, (3,), dtype=torch.int64)
>>> loss = F.cross_entropy(input, target)
>>> loss.backward()
>>>
>>> # Example of target with class probabilities
>>> input = torch.randn(3, 5, requires_grad=True)
>>> target = torch.randn(3, 5).softmax(dim=1)
>>> loss = F.cross_entropy(input, target)
>>> loss.backward()

```

