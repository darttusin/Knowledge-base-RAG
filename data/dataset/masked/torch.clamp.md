torch.clamp 
==========================================================

torch. clamp ( *input*  , *min = None*  , *max = None*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Clamps all elements in `input`  into the range *[*[`min`](torch.min.html#torch.min "torch.min")  , [`max`](torch.max.html#torch.max "torch.max") *]* .
Letting min_value and max_value be [`min`](torch.min.html#torch.min "torch.min")  and [`max`](torch.max.html#torch.max "torch.max")  , respectively, this returns: 

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
<mi>
            min
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<mi>
            max
           </mi>
<mo>
            ⁡
           </mo>
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
<mo separator="true">
            ,
           </mo>
<msub>
<mtext>
             min_value
            </mtext>
<mi>
             i
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
<mo separator="true">
            ,
           </mo>
<msub>
<mtext>
             max_value
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
           y_i = min(max(x_i, text{min_value}_i), text{max_value}_i)
          </annotation>
</semantics>
</math> -->
y i = min ⁡ ( max ⁡ ( x i , min_value i ) , max_value i ) y_i = min(max(x_i, text{min_value}_i), text{max_value}_i)

y i ​ = min ( max ( x i ​ , min_value i ​ ) , max_value i ​ )

If [`min`](torch.min.html#torch.min "torch.min")  is `None`  , there is no lower bound.
Or, if [`max`](torch.max.html#torch.max "torch.max")  is `None`  there is no upper bound. 

Note 

If [`min`](torch.min.html#torch.min "torch.min")  is greater than [`max`](torch.max.html#torch.max "torch.max") [`torch.clamp(..., min, max)`](#torch.clamp "torch.clamp")  sets all elements in `input`  to the value of [`max`](torch.max.html#torch.max "torch.max")  .

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **min** ( *Number* *or* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – lower-bound of the range to be clamped to
* **max** ( *Number* *or* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – upper-bound of the range to be clamped to

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.randn(4)
>>> a
tensor([-1.7120,  0.1734, -0.0478, -0.0922])
>>> torch.clamp(a, min=-0.5, max=0.5)
tensor([-0.5000,  0.1734, -0.0478, -0.0922])

>>> min = torch.linspace(-1, 1, steps=4)
>>> torch.clamp(a, min=min)
tensor([-1.0000,  0.1734,  0.3333,  1.0000])

```

