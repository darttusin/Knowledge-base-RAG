AlphaDropout 
============================================================

*class* torch.nn. AlphaDropout ( *p = 0.5*  , *inplace = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/dropout.py#L215) 
:   Applies Alpha Dropout over the input. 

Alpha Dropout is a type of Dropout that maintains the self-normalizing
property.
For an input with zero mean and unit standard deviation, the output of
Alpha Dropout maintains the original mean and standard deviation of the
input.
Alpha Dropout goes hand-in-hand with SELU activation function, which ensures
that the outputs have zero mean and unit standard deviation. 

During training, it randomly masks some of the elements of the input
tensor with probability *p*  using samples from a bernoulli distribution.
The elements to masked are randomized on every forward call, and scaled
and shifted to maintain zero mean and unit standard deviation. 

During evaluation the module simply computes an identity function. 

More details can be found in the paper [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)  . 

Parameters
:   * **p** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – probability of an element to be dropped. Default: 0.5
* **inplace** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If set to `True`  , will do this operation
in-place

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
</math> -->( ∗ ) (*)( ∗ )  . Input can be of any shape

* Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( ∗ ) (*)( ∗ )  . Output is of the same shape as input

Examples: 

```
>>> m = nn.AlphaDropout(p=0.2)
>>> input = torch.randn(20, 16)
>>> output = m(input)

```

