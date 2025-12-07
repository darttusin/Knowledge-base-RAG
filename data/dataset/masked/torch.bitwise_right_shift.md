torch.bitwise_right_shift 
========================================================================================

torch. bitwise_right_shift ( *input*  , *other*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the right arithmetic shift of `input`  by `other`  bits.
The input tensor must be of integral type. This operator supports [broadcasting to a common shape](../notes/broadcasting.html#broadcasting-semantics)  and [type promotion](../tensor_attributes.html#type-promotion-doc)  .
In any case, if the value of the right operand is negative or is greater
or equal to the number of bits in the promoted left operand, the behavior is undefined. 

The operation applied is: 

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
            &gt;
           </mo>
<mo>
            &gt;
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
           text{out}_i = text{input}_i &gt;&gt; text{other}_i
          </annotation>
</semantics>
</math> -->
out i = input i > > other i text{out}_i = text{input}_i >> text{other}_i

out i ​ = input i ​ >> other i ​

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *or* *Scalar*  ) – the first input tensor
* **other** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *or* *Scalar*  ) – the second input tensor

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.bitwise_right_shift(torch.tensor([-2, -7, 31], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
tensor([-1, -7,  3], dtype=torch.int8)

```

