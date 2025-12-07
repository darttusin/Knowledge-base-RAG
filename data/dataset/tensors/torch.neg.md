torch.neg 
======================================================

torch. neg ( *input*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a new tensor with the negative of the elements of `input`  . 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            out
           </mtext>
<mo>
            =
           </mo>
<mo>
            −
           </mo>
<mn>
            1
           </mn>
<mo>
            ×
           </mo>
<mtext>
            input
           </mtext>
</mrow>
<annotation encoding="application/x-tex">
           text{out} = -1 times text{input}
          </annotation>
</semantics>
</math> -->
out = − 1 × input text{out} = -1 times text{input}

out = − 1 × input

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.randn(5)
>>> a
tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])
>>> torch.neg(a)
tensor([-0.0090,  0.2262,  0.0682,  0.2866, -0.3940])

```

