torch.nn.functional.cosine_similarity 
===============================================================================================================

torch.nn.functional. cosine_similarity ( *x1*  , *x2*  , *dim = 1*  , *eps = 1e-8* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns cosine similarity between `x1`  and `x2`  , computed along dim. `x1`  and `x2`  must be broadcastable
to a common shape. `dim`  refers to the dimension in this common shape. Dimension `dim`  of the output is
squeezed (see [`torch.squeeze()`](torch.squeeze.html#torch.squeeze "torch.squeeze")  ), resulting in the
output tensor having 1 fewer dimension. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            similarity
           </mtext>
<mo>
            =
           </mo>
<mfrac>
<mrow>
<msub>
<mi>
               x
              </mi>
<mn>
               1
              </mn>
</msub>
<mo>
              ⋅
             </mo>
<msub>
<mi>
               x
              </mi>
<mn>
               2
              </mn>
</msub>
</mrow>
<mrow>
<mi>
              max
             </mi>
<mo>
              ⁡
             </mo>
<mo stretchy="false">
              (
             </mo>
<mi mathvariant="normal">
              ∥
             </mi>
<msub>
<mi>
               x
              </mi>
<mn>
               1
              </mn>
</msub>
<msub>
<mi mathvariant="normal">
               ∥
              </mi>
<mn>
               2
              </mn>
</msub>
<mo separator="true">
              ,
             </mo>
<mi>
              ϵ
             </mi>
<mo stretchy="false">
              )
             </mo>
<mo>
              ⋅
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
<mi mathvariant="normal">
              ∥
             </mi>
<msub>
<mi>
               x
              </mi>
<mn>
               2
              </mn>
</msub>
<msub>
<mi mathvariant="normal">
               ∥
              </mi>
<mn>
               2
              </mn>
</msub>
<mo separator="true">
              ,
             </mo>
<mi>
              ϵ
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           text{similarity} = dfrac{x_1 cdot x_2}{max(Vert x_1 Vert _2, epsilon) cdot max(Vert x_2 Vert _2, epsilon)}
          </annotation>
</semantics>
</math> -->
similarity = x 1 ⋅ x 2 max ⁡ ( ∥ x 1 ∥ 2 , ϵ ) ⋅ max ⁡ ( ∥ x 2 ∥ 2 , ϵ ) text{similarity} = dfrac{x_1 cdot x_2}{max(Vert x_1 Vert _2, epsilon) cdot max(Vert x_2 Vert _2, epsilon)}

similarity = max ( ∥ x 1 ​ ∥ 2 ​ , ϵ ) ⋅ max ( ∥ x 2 ​ ∥ 2 ​ , ϵ ) x 1 ​ ⋅ x 2 ​ ​

Supports [type promotion](../tensor_attributes.html#type-promotion-doc)  . 

Parameters
:   * **x1** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – First input.
* **x2** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Second input.
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – Dimension along which cosine similarity is computed. Default: 1
* **eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – Small value to avoid division by zero.
Default: 1e-8

Example: 

```
>>> input1 = torch.randn(100, 128)
>>> input2 = torch.randn(100, 128)
>>> output = F.cosine_similarity(input1, input2)
>>> print(output)

```

