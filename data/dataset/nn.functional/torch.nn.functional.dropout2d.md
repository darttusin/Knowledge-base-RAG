torch.nn.functional.dropout2d 
==============================================================================================

torch.nn.functional. dropout2d ( *input*  , *p = 0.5*  , *training = True*  , *inplace = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L1500) 
:   Randomly zero out entire channels (a channel is a 2D feature map). 

For example, the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->j jj  -th channel of the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->i ii  -th sample in the
batched input is a 2D tensor <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            input
           </mtext>
<mo stretchy="false">
            [
           </mo>
<mi>
            i
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            j
           </mi>
<mo stretchy="false">
            ]
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{input}[i, j]
          </annotation>
</semantics>
</math> -->input [ i , j ] text{input}[i, j]input [ i , j ]  of the input tensor.
Each channel will be zeroed out independently on every forward call with
probability `p`  using samples from a Bernoulli distribution. 

See [`Dropout2d`](torch.nn.Dropout2d.html#torch.nn.Dropout2d "torch.nn.Dropout2d")  for details. 

Parameters
:   * **p** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – probability of a channel to be zeroed. Default: 0.5
* **training** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – apply dropout if is `True`  . Default: `True`
* **inplace** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If set to `True`  , will do this operation in-place. Default: `False`

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

