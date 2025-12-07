torch.nn.functional.mish 
====================================================================================

torch.nn.functional. mish ( *input*  , *inplace = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L2378) 
:   Apply the Mish function, element-wise. 

Mish: A Self Regularized Non-Monotonic Neural Activation Function. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            Mish
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mi>
            x
           </mi>
<mo>
            ∗
           </mo>
<mtext>
            Tanh
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mtext>
            Softplus
           </mtext>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{Mish}(x) = x * text{Tanh}(text{Softplus}(x))
          </annotation>
</semantics>
</math> -->
Mish ( x ) = x ∗ Tanh ( Softplus ( x ) ) text{Mish}(x) = x * text{Tanh}(text{Softplus}(x))

Mish ( x ) = x ∗ Tanh ( Softplus ( x ))

Note 

See [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681)

See [`Mish`](torch.nn.Mish.html#torch.nn.Mish "torch.nn.Mish")  for more details. 

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

