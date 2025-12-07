torch.addcmul 
==============================================================

torch. addcmul ( *input*  , *tensor1*  , *tensor2*  , *** , *value = 1*  , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Performs the element-wise multiplication of `tensor1`  by `tensor2`  , multiplies the result by the scalar `value`  and adds it to `input`  . 

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
<msub>
<mtext>
             tensor1
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo>
            ×
           </mo>
<msub>
<mtext>
             tensor2
            </mtext>
<mi>
             i
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           text{out}_i = text{input}_i + text{value} times text{tensor1}_i times text{tensor2}_i
          </annotation>
</semantics>
</math> -->
out i = input i + value × tensor1 i × tensor2 i text{out}_i = text{input}_i + text{value} times text{tensor1}_i times text{tensor2}_i

out i ​ = input i ​ + value × tensor1 i ​ × tensor2 i ​

The shapes of [`tensor`](torch.tensor.html#torch.tensor "torch.tensor")  , `tensor1`  , and `tensor2`  must be [broadcastable](../notes/broadcasting.html#broadcasting-semantics)  . 

For inputs of type *FloatTensor* or *DoubleTensor* , `value`  must be
a real number, otherwise an integer. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the tensor to be added
* **tensor1** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the tensor to be multiplied
* **tensor2** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the tensor to be multiplied

Keyword Arguments
:   * **value** ( *Number* *,* *optional*  ) – multiplier for <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                t
               </mi>
<mi>
                e
               </mi>
<mi>
                n
               </mi>
<mi>
                s
               </mi>
<mi>
                o
               </mi>
<mi>
                r
               </mi>
<mn>
                1.
               </mn>
<mo>
                ∗
               </mo>
<mi>
                t
               </mi>
<mi>
                e
               </mi>
<mi>
                n
               </mi>
<mi>
                s
               </mi>
<mi>
                o
               </mi>
<mi>
                r
               </mi>
<mn>
                2
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               tensor1 .* tensor2
              </annotation>
</semantics>
</math> -->t e n s o r 1. ∗ t e n s o r 2 tensor1 .* tensor2t e n sor 1. ∗ t e n sor 2

* **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> t = torch.randn(1, 3)
>>> t1 = torch.randn(3, 1)
>>> t2 = torch.randn(1, 3)
>>> torch.addcmul(t, t1, t2, value=0.1)
tensor([[-0.8635, -0.6391,  1.6174],
        [-0.7617, -0.5879,  1.7388],
        [-0.8353, -0.6249,  1.6511]])

```

