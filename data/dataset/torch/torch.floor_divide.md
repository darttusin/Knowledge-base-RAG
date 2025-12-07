torch.floor_divide 
=========================================================================

torch. floor_divide ( *input*  , *other*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Note 

Before PyTorch 1.13 [`torch.floor_divide()`](#torch.floor_divide "torch.floor_divide")  incorrectly performed
truncation division. To restore the previous behavior use [`torch.div()`](torch.div.html#torch.div "torch.div")  with `rounding_mode='trunc'`  .

Computes `input`  divided by `other`  , elementwise, and floors
the result. 

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
<mtext>
            floor
           </mtext>
<mrow>
<mo fence="true">
             (
            </mo>
<mfrac>
<msub>
<mtext>
               input
              </mtext>
<mi>
               i
              </mi>
</msub>
<msub>
<mtext>
               other
              </mtext>
<mi>
               i
              </mi>
</msub>
</mfrac>
<mo fence="true">
             )
            </mo>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           text{{out}}_i = text{floor} left( frac{{text{{input}}_i}}{{text{{other}}_i}} right)
          </annotation>
</semantics>
</math> -->
out i = floor ( input i other i ) text{{out}}_i = text{floor} left( frac{{text{{input}}_i}}{{text{{other}}_i}} right)

out i ​ = floor ( other i ​ input i ​ ​ )

Supports broadcasting to a common shape, type promotion, and integer and float inputs. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *or* *Number*  ) – the dividend
* **other** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *or* *Number*  ) – the divisor

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.tensor([4.0, 3.0])
>>> b = torch.tensor([2.0, 2.0])
>>> torch.floor_divide(a, b)
tensor([2.0, 1.0])
>>> torch.floor_divide(a, 1.4)
tensor([2.0, 2.0])

```

