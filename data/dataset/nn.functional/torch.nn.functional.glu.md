torch.nn.functional.glu 
==================================================================================

torch.nn.functional. glu ( *input*  , *dim = -1* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L1715) 
:   The gated linear unit. Computes: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            GLU
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mi>
            a
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            b
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mi>
            a
           </mi>
<mo>
            ⊗
           </mo>
<mi>
            σ
           </mi>
<mo stretchy="false">
            (
           </mo>
<mi>
            b
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{GLU}(a, b) = a otimes sigma(b)
          </annotation>
</semantics>
</math> -->
GLU ( a , b ) = a ⊗ σ ( b ) text{GLU}(a, b) = a otimes sigma(b)

GLU ( a , b ) = a ⊗ σ ( b )

where *input* is split in half along *dim* to form *a* and *b* , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            σ
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           sigma
          </annotation>
</semantics>
</math> -->σ sigmaσ  is the sigmoid function and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
            ⊗
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           otimes
          </annotation>
</semantics>
</math> -->⊗ otimes⊗  is the element-wise product between matrices. 

See [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083)  . 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – input tensor
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – dimension on which to split the input. Default: -1

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

