torch.nn.functional.softmax 
==========================================================================================

torch.nn.functional. softmax ( *input*  , *dim = None*  , *_stacklevel = 3*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L2100) 
:   Apply a softmax function. 

Softmax is defined as: 

<!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            Softmax
           </mtext>
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             x
            </mi>
<mi>
             i
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
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
<msub>
<mi>
               x
              </mi>
<mi>
               i
              </mi>
</msub>
<mo stretchy="false">
              )
             </mo>
</mrow>
<mrow>
<msub>
<mo>
               ∑
              </mo>
<mi>
               j
              </mi>
</msub>
<mi>
              exp
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
<mi>
               j
              </mi>
</msub>
<mo stretchy="false">
              )
             </mo>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           text{Softmax}(x_{i}) = frac{exp(x_i)}{sum_j exp(x_j)}
          </annotation>
</semantics>
</math> -->Softmax ( x i ) = exp ⁡ ( x i ) ∑ j exp ⁡ ( x j ) text{Softmax}(x_{i}) = frac{exp(x_i)}{sum_j exp(x_j)}Softmax ( x i ​ ) = ∑ j ​ e x p ( x j ​ ) e x p ( x i ​ ) ​ 

It is applied to all slices along dim, and will re-scale them so that the elements
lie in the range *[0, 1]* and sum to 1. 

See [`Softmax`](torch.nn.Softmax.html#torch.nn.Softmax "torch.nn.Softmax")  for more details. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – input
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – A dimension along which softmax will be computed.
* **dtype** ( [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired data type of returned tensor.
If specified, the input tensor is casted to `dtype`  before the operation
is performed. This is useful for preventing data type overflows. Default: None.

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

Note 

This function doesn’t work directly with NLLLoss,
which expects the Log to be computed between the Softmax and itself.
Use log_softmax instead (it’s faster and has better numerical properties).

