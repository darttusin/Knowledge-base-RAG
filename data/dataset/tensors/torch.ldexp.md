torch.ldexp 
==========================================================

torch. ldexp ( *input*  , *other*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Multiplies `input`  by 2 ** `other`  . 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mtext>
             out
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo>
            =
           </mo>
<msub>
<mtext>
             input
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo>
            ∗
           </mo>
<msubsup>
<mn>
             2
            </mn>
<mi>
             i
            </mi>
<mtext>
             other
            </mtext>
</msubsup>
</mrow>
<annotation encoding="application/x-tex">
           text{{out}}_i = text{{input}}_i * 2^text{{other}}_i
          </annotation>
</semantics>
</math> -->
out i = input i ∗ 2 i other text{{out}}_i = text{{input}}_i * 2^text{{other}}_i

out i ​ = input i ​ ∗ 2 i other ​

Typically this function is used to construct floating point numbers by multiplying
mantissas in `input`  with integral powers of two created from the exponents
in `other`  . 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **other** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – a tensor of exponents, typically integers.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.ldexp(torch.tensor([1.]), torch.tensor([1]))
tensor([2.])
>>> torch.ldexp(torch.tensor([1.0]), torch.tensor([1, 2, 3, 4]))
tensor([ 2.,  4.,  8., 16.])

```

