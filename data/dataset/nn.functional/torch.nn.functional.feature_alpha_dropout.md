torch.nn.functional.feature_alpha_dropout 
========================================================================================================================

torch.nn.functional. feature_alpha_dropout ( *input*  , *p = 0.5*  , *training = False*  , *inplace = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L1611) 
:   Randomly masks out entire channels (a channel is a feature map). 

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
</math> -->i ii  -th sample in the batch input
is a tensor <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->input [ i , j ] text{input}[i, j]input [ i , j ]  of the input tensor. Instead of
setting activations to zero, as in regular Dropout, the activations are set
to the negative saturation value of the SELU activation function. 

Each element will be masked independently on every forward call with
probability `p`  using samples from a Bernoulli distribution.
The elements to be masked are randomized on every forward call, and scaled
and shifted to maintain zero mean and unit variance. 

See [`FeatureAlphaDropout`](torch.nn.FeatureAlphaDropout.html#torch.nn.FeatureAlphaDropout "torch.nn.FeatureAlphaDropout")  for details. 

Parameters
:   * **p** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – dropout probability of a channel to be zeroed. Default: 0.5
* **training** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – apply dropout if is `True`  . Default: `True`
* **inplace** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If set to `True`  , will do this operation in-place. Default: `False`

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

