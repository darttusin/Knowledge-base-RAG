torch.log1p 
==========================================================

torch. log1p ( *input*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a new tensor with the natural logarithm of (1 + `input`  ). 

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
<msub>
<mrow>
<mi>
              log
             </mi>
<mo>
              ⁡
             </mo>
</mrow>
<mi>
             e
            </mi>
</msub>
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             x
            </mi>
<mi>
             i
            </mi>
</msub>
<mo>
            +
           </mo>
<mn>
            1
           </mn>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           y_i = log_{e} (x_i + 1)
          </annotation>
</semantics>
</math> -->
y i = log ⁡ e ( x i + 1 ) y_i = log_{e} (x_i + 1)

y i ​ = lo g e ​ ( x i ​ + 1 )

Note 

This function is more accurate than [`torch.log()`](torch.log.html#torch.log "torch.log")  for small
values of `input`

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.randn(5)
>>> a
tensor([-1.0090, -0.9923,  1.0249, -0.5372,  0.2492])
>>> torch.log1p(a)
tensor([    nan, -4.8653,  0.7055, -0.7705,  0.2225])

```

