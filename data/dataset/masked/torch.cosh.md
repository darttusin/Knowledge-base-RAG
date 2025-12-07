torch.cosh 
========================================================

torch. cosh ( *input*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a new tensor with the hyperbolic cosine of the elements of `input`  . 

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
            cosh
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
           text{out}_{i} = cosh(text{input}_{i})
          </annotation>
</semantics>
</math> -->
out i = cosh ⁡ ( input i ) text{out}_{i} = cosh(text{input}_{i})

out i ​ = cosh ( input i ​ )

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.randn(4)
>>> a
tensor([ 0.1632,  1.1835, -0.6979, -0.7325])
>>> torch.cosh(a)
tensor([ 1.0133,  1.7860,  1.2536,  1.2805])

```

Note 

When `input`  is on the CPU, the implementation of torch.cosh may use
the Sleef library, which rounds very large results to infinity or negative
infinity. See [here](https://sleef.org/purec.xhtml)  for details.

