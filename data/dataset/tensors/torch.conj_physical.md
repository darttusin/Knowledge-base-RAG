torch.conj_physical 
===========================================================================

torch. conj_physical ( *input*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the element-wise conjugate of the given `input`  tensor.
If `input`  has a non-complex dtype, this function just returns `input`  . 

Note 

This performs the conjugate operation regardless of the fact conjugate bit is set or not.

Warning 

In the future, [`torch.conj_physical()`](#torch.conj_physical "torch.conj_physical")  may return a non-writeable view for an `input`  of
non-complex dtype. It’s recommended that programs not modify the tensor returned by [`torch.conj_physical()`](#torch.conj_physical "torch.conj_physical")  when `input`  is of non-complex dtype to be compatible with this change.

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
            c
           </mi>
<mi>
            o
           </mi>
<mi>
            n
           </mi>
<mi>
            j
           </mi>
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
           text{out}_{i} = conj(text{input}_{i})
          </annotation>
</semantics>
</math> -->
out i = c o n j ( input i ) text{out}_{i} = conj(text{input}_{i})

out i ​ = co nj ( input i ​ )

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> torch.conj_physical(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]))
tensor([-1 - 1j, -2 - 2j, 3 + 3j])

```

