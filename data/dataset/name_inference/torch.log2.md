torch.log2 
========================================================

torch. log2 ( *input : [Tensor](../tensors.html#torch.Tensor "torch.Tensor")*  , *** , *out : Optional [ [Tensor](../tensors.html#torch.Tensor "torch.Tensor") ]* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a new tensor with the logarithm to the base 2 of the elements
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
<mn>
             2
            </mn>
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
           y_{i} = log_{2} (x_{i})
          </annotation>
</semantics>
</math> -->
y i = log ⁡ 2 ( x i ) y_{i} = log_{2} (x_{i})

y i ​ = lo g 2 ​ ( x i ​ )

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.

Keyword Arguments
: **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Example: 

```
>>> a = torch.rand(5)
>>> a
tensor([ 0.8419,  0.8003,  0.9971,  0.5287,  0.0490])

>>> torch.log2(a)
tensor([-0.2483, -0.3213, -0.0042, -0.9196, -4.3504])

```

