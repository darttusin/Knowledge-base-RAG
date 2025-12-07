torch.sub 
======================================================

torch. sub ( *input*  , *other*  , *** , *alpha = 1*  , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Subtracts `other`  , scaled by `alpha`  , from `input`  . 

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
            −
           </mo>
<mtext>
            alpha
           </mtext>
<mo>
            ×
           </mo>
<msub>
<mtext>
             other
            </mtext>
<mi>
             i
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           text{{out}}_i = text{{input}}_i - text{{alpha}} times text{{other}}_i
          </annotation>
</semantics>
</math> -->
out i = input i − alpha × other i text{{out}}_i = text{{input}}_i - text{{alpha}} times text{{other}}_i

out i ​ = input i ​ − alpha × other i ​

Supports [broadcasting to a common shape](../notes/broadcasting.html#broadcasting-semantics)  , [type promotion](../tensor_attributes.html#type-promotion-doc)  , and integer, float, and complex inputs. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **other** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *or* *Number*  ) – the tensor or number to subtract from `input`  .

Keyword Arguments
:   * **alpha** ( *Number*  ) – the multiplier for `other`  .
* **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.tensor((1, 2))
>>> b = torch.tensor((0, 1))
>>> torch.sub(a, b, alpha=2)
tensor([1, 0])

```

