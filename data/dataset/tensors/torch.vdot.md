torch.vdot 
========================================================

torch. vdot ( *input*  , *other*  , *** , *out = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the dot product of two 1D vectors along a dimension. 

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

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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

Note 

Unlike NumPy’s vdot, torch.vdot intentionally only supports computing the dot product
of two 1D tensors with the same number of elements.

See also 

[`torch.linalg.vecdot()`](torch.linalg.vecdot.html#torch.linalg.vecdot "torch.linalg.vecdot")  computes the dot product of two batches of vectors along a dimension.

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – first tensor in the dot product, must be 1D. Its conjugate is used if it’s complex.
* **other** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – second tensor in the dot product, must be 1D.

Keyword args: 

Note 

out (Tensor, optional): the output tensor.

Example: 

```
>>> torch.vdot(torch.tensor([2, 3]), torch.tensor([2, 1]))
tensor(7)
>>> a = torch.tensor((1 +2j, 3 - 1j))
>>> b = torch.tensor((2 +1j, 4 - 0j))
>>> torch.vdot(a, b)
tensor([16.+1.j])
>>> torch.vdot(b, a)
tensor([16.-1.j])

```

