torch.log 
======================================================

torch. log ( *input*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a new tensor with the natural logarithm of the elements
of `input`  . 

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
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           y_{i} = log_{e} (x_{i})
          </annotation>
</semantics>
</math> -->
y i = log ⁡ e ( x i ) y_{i} = log_{e} (x_{i})

y i ​ = lo g e ​ ( x i ​ )

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.rand(5) * 5
>>> a
tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
>>> torch.log(a)
tensor([ 1.5637,  1.4640,  0.1952, -1.4226,  1.5204])

```

