torch.linalg.vecdot 
==========================================================================

torch.linalg. vecdot ( *x*  , *y*  , *** , *dim = -1*  , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the dot product of two batches of vectors along a dimension. 

In symbols, this function computes 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<munderover>
<mo>
             ∑
            </mo>
<mrow>
<mi>
              i
             </mi>
<mo>
              =
             </mo>
<mn>
              1
             </mn>
</mrow>
<mi>
             n
            </mi>
</munderover>
<mover accent="true">
<msub>
<mi>
              x
             </mi>
<mi>
              i
             </mi>
</msub>
<mo stretchy="true">
             ‾
            </mo>
</mover>
<msub>
<mi>
             y
            </mi>
<mi>
             i
            </mi>
</msub>
<mi mathvariant="normal">
            .
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           sum_{i=1}^n overline{x_i}y_i.
          </annotation>
</semantics>
</math> -->
∑ i = 1 n x i ‾ y i . sum_{i=1}^n overline{x_i}y_i.

i = 1 ∑ n ​ x i ​ ​ y i ​ .

over the dimension `dim`  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mover accent="true">
<msub>
<mi>
              x
             </mi>
<mi>
              i
             </mi>
</msub>
<mo stretchy="true">
             ‾
            </mo>
</mover>
</mrow>
<annotation encoding="application/x-tex">
           overline{x_i}
          </annotation>
</semantics>
</math> -->x i ‾ overline{x_i}x i ​ ​  denotes the conjugate for complex
vectors, and it is the identity for real vectors. 

Supports input of half, bfloat16, float, double, cfloat, cdouble and integral dtypes.
It also supports broadcasting. 

Parameters
:   * **x** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – first batch of vectors of shape *(*, n)* .
* **y** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – second batch of vectors of shape *(*, n)* .

Keyword Arguments
:   * **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Dimension along which to compute the dot product. Default: *-1* .
* **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – output tensor. Ignored if *None* . Default: *None* .

Examples: 

```
>>> v1 = torch.randn(3, 2)
>>> v2 = torch.randn(3, 2)
>>> linalg.vecdot(v1, v2)
tensor([ 0.3223,  0.2815, -0.1944])
>>> torch.vdot(v1[0], v2[0])
tensor(0.3223)

```

