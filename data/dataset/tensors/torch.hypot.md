torch.hypot 
==========================================================

torch. hypot ( *input*  , *other*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Given the legs of a right triangle, return its hypotenuse. 

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
<msqrt>
<mrow>
<msubsup>
<mtext>
               input
              </mtext>
<mi>
               i
              </mi>
<mn>
               2
              </mn>
</msubsup>
<mo>
              +
             </mo>
<msubsup>
<mtext>
               other
              </mtext>
<mi>
               i
              </mi>
<mn>
               2
              </mn>
</msubsup>
</mrow>
</msqrt>
</mrow>
<annotation encoding="application/x-tex">
           text{out}_{i} = sqrt{text{input}_{i}^{2} + text{other}_{i}^{2}}
          </annotation>
</semantics>
</math> -->
out i = input i 2 + other i 2 text{out}_{i} = sqrt{text{input}_{i}^{2} + text{other}_{i}^{2}}

out i ​ = input i 2 ​ + other i 2 ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuODhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTk0NCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTgzIDkwCmwwIC0wCmM0LC02LjcsMTAsLTEwLDE4LC0xMCBINDAwMDAwdjQwCkgxMDEzLjFzLTgzLjQsMjY4LC0yNjQuMSw4NDBjLTE4MC43LDU3MiwtMjc3LDg3Ni4zLC0yODksOTEzYy00LjcsNC43LC0xMi43LDcsLTI0LDcKcy0xMiwwLC0xMiwwYy0xLjMsLTMuMywtMy43LC0xMS43LC03LC0yNWMtMzUuMywtMTI1LjMsLTEwNi43LC0zNzMuMywtMjE0LC03NDQKYy0xMCwxMiwtMjEsMjUsLTMzLDM5cy0zMiwzOSwtMzIsMzljLTYsLTUuMywtMTUsLTE0LC0yNywtMjZzMjUsLTMwLDI1LC0zMApjMjYuNywtMzIuNyw1MiwtNjMsNzYsLTkxczUyLC02MCw1MiwtNjBzMjA4LDcyMiwyMDgsNzIyCmM1NiwtMTc1LjMsMTI2LjMsLTM5Ny4zLDIxMSwtNjY2Yzg0LjcsLTI2OC43LDE1My44LC00ODguMiwyMDcuNSwtNjU4LjUKYzUzLjcsLTE3MC4zLDg0LjUsLTI2Ni44LDkyLjUsLTI4OS41egpNMTAwMSA4MGg0MDAwMDB2NDBoLTQwMDAwMHoiPgo8L3BhdGg+Cjwvc3ZnPg==)​

The shapes of `input`  and `other`  must be [broadcastable](../notes/broadcasting.html#broadcasting-semantics)  . 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the first input tensor
* **other** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the second input tensor

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.hypot(torch.tensor([4.0]), torch.tensor([3.0, 4.0, 5.0]))
tensor([5.0000, 5.6569, 6.4031])

```

