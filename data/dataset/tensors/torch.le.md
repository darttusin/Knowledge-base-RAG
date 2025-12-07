torch.le 
====================================================

torch. le ( *input*  , *other*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            input
           </mtext>
<mo>
            ≤
           </mo>
<mtext>
            other
           </mtext>
</mrow>
<annotation encoding="application/x-tex">
           text{input} leq text{other}
          </annotation>
</semantics>
</math> -->input ≤ other text{input} leq text{other}input ≤ other  element-wise. 

The second argument can be a number or a tensor whose shape is [broadcastable](../notes/broadcasting.html#broadcasting-semantics)  with the first argument. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the tensor to compare
* **other** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *or* *Scalar*  ) – the tensor or value to compare

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Returns
:   A boolean tensor that is True where `input`  is less than or equal to `other`  and False elsewhere

Example: 

```
>>> torch.le(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
tensor([[True, False], [True, True]])

```

