torch.sinh 
========================================================

torch. sinh ( *input*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a new tensor with the hyperbolic sine of the elements of `input`  . 

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
<mi>
            sinh
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<msub>
<mtext>
             input
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{out}_{i} = sinh(text{input}_{i})
          </annotation>
</semantics>
</math> -->
out i = sinh ⁡ ( input i ) text{out}_{i} = sinh(text{input}_{i})

out i ​ = sinh ( input i ​ )

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.randn(4)
>>> a
tensor([ 0.5380, -0.8632, -0.1265,  0.9399])
>>> torch.sinh(a)
tensor([ 0.5644, -0.9744, -0.1268,  1.0845])

```

Note 

When `input`  is on the CPU, the implementation of torch.sinh may use
the Sleef library, which rounds very large results to infinity or negative
infinity. See [here](https://sleef.org/purec.xhtml)  for details.

