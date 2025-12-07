torch.floor 
==========================================================

torch. floor ( *input*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a new tensor with the floor of the elements of `input`  ,
the largest integer less than or equal to each element. 

For integer inputs, follows the array-api convention of returning a
copy of the input tensor. 

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
<mrow>
<mo fence="true">
             ⌊
            </mo>
<msub>
<mtext>
              input
             </mtext>
<mi>
              i
             </mi>
</msub>
<mo fence="true">
             ⌋
            </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           text{out}_{i} = leftlfloor text{input}_{i} rightrfloor
          </annotation>
</semantics>
</math> -->
out i = ⌊ input i ⌋ text{out}_{i} = leftlfloor text{input}_{i} rightrfloor

out i ​ = ⌊ input i ​ ⌋

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.randn(4)
>>> a
tensor([-0.8166,  1.5308, -0.2530, -0.2091])
>>> torch.floor(a)
tensor([-1.,  1., -1., -1.])

```

