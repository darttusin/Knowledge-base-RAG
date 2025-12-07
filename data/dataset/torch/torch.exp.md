torch.exp 
======================================================

torch. exp ( *input*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a new tensor with the exponential of the elements
of the input tensor `input`  . 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             y
            </mi>
<mi>
             i
            </mi>
</msub>
<mo>
            =
           </mo>
<msup>
<mi>
             e
            </mi>
<msub>
<mi>
              x
             </mi>
<mi>
              i
             </mi>
</msub>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           y_{i} = e^{x_{i}}
          </annotation>
</semantics>
</math> -->
y i = e x i y_{i} = e^{x_{i}}

y i ​ = e x i ​

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.exp(torch.tensor([0, math.log(2.)]))
tensor([ 1.,  2.])

```

