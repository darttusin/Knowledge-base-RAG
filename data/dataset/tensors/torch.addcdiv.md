torch.addcdiv 
==============================================================

torch. addcdiv ( *input*  , *tensor1*  , *tensor2*  , *** , *value = 1*  , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Performs the element-wise division of `tensor1`  by `tensor2`  ,
multiplies the result by the scalar `value`  and adds it to `input`  . 

Warning 

Integer division with addcdiv is no longer supported, and in a future
release addcdiv will perform a true division of tensor1 and tensor2.
The historic addcdiv behavior can be implemented as
(input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype)
for integer inputs and as (input + value * tensor1 / tensor2) for float inputs.
The future addcdiv behavior is just the latter implementation:
(input + value * tensor1 / tensor2), for all dtypes.

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
            +
           </mo>
<mtext>
            value
           </mtext>
<mo>
            ×
           </mo>
<mfrac>
<msub>
<mtext>
              tensor1
             </mtext>
<mi>
              i
             </mi>
</msub>
<msub>
<mtext>
              tensor2
             </mtext>
<mi>
              i
             </mi>
</msub>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           text{out}_i = text{input}_i + text{value} times frac{text{tensor1}_i}{text{tensor2}_i}
          </annotation>
</semantics>
</math> -->
out i = input i + value × tensor1 i tensor2 i text{out}_i = text{input}_i + text{value} times frac{text{tensor1}_i}{text{tensor2}_i}

out i ​ = input i ​ + value × tensor2 i ​ tensor1 i ​ ​

The shapes of `input`  , `tensor1`  , and `tensor2`  must be [broadcastable](../notes/broadcasting.html#broadcasting-semantics)  . 

For inputs of type *FloatTensor* or *DoubleTensor* , `value`  must be
a real number, otherwise an integer. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the tensor to be added
* **tensor1** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the numerator tensor
* **tensor2** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the denominator tensor

Keyword Arguments
:   * **value** ( *Number* *,* *optional*  ) – multiplier for <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
                tensor1
               </mtext>
<mi mathvariant="normal">
                /
               </mi>
<mtext>
                tensor2
               </mtext>
</mrow>
<annotation encoding="application/x-tex">
               text{tensor1} / text{tensor2}
              </annotation>
</semantics>
</math> -->tensor1 / tensor2 text{tensor1} / text{tensor2}tensor1 / tensor2

* **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> t = torch.randn(1, 3)
>>> t1 = torch.randn(3, 1)
>>> t2 = torch.randn(1, 3)
>>> torch.addcdiv(t, t1, t2, value=0.1)
tensor([[-0.2312, -3.6496,  0.1312],
        [-1.0428,  3.4292, -0.1030],
        [-0.5369, -0.9829,  0.0430]])

```

