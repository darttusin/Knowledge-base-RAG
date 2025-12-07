torch.quantized_batch_norm 
==========================================================================================

torch. quantized_batch_norm ( *input*  , *weight=None*  , *bias=None*  , *mean*  , *var*  , *eps*  , *output_scale*  , *output_zero_point* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Applies batch normalization on a 4D (NCHW) quantized tensor. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            y
           </mi>
<mo>
            =
           </mo>
<mfrac>
<mrow>
<mi>
              x
             </mi>
<mo>
              −
             </mo>
<mi mathvariant="normal">
              E
             </mi>
<mo stretchy="false">
              [
             </mo>
<mi>
              x
             </mi>
<mo stretchy="false">
              ]
             </mo>
</mrow>
<msqrt>
<mrow>
<mrow>
<mi mathvariant="normal">
                V
               </mi>
<mi mathvariant="normal">
                a
               </mi>
<mi mathvariant="normal">
                r
               </mi>
</mrow>
<mo stretchy="false">
               [
              </mo>
<mi>
               x
              </mi>
<mo stretchy="false">
               ]
              </mo>
<mo>
               +
              </mo>
<mi>
               ϵ
              </mi>
</mrow>
</msqrt>
</mfrac>
<mo>
            ∗
           </mo>
<mi>
            γ
           </mi>
<mo>
            +
           </mo>
<mi>
            β
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           y = frac{x - mathrm{E}[x]}{sqrt{mathrm{Var}[x] + epsilon}} * gamma + beta
          </annotation>
</semantics>
</math> -->
y = x − E [ x ] V a r [ x ] + ϵ ∗ γ + β y = frac{x - mathrm{E}[x]}{sqrt{mathrm{Var}[x] + epsilon}} * gamma + beta

y = Var [ x ] + ϵ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMjhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTI5NiIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNMjYzLDY4MWMwLjcsMCwxOCwzOS43LDUyLDExOQpjMzQsNzkuMyw2OC4xNjcsMTU4LjcsMTAyLjUsMjM4YzM0LjMsNzkuMyw1MS44LDExOS4zLDUyLjUsMTIwCmMzNDAsLTcwNC43LDUxMC43LC0xMDYwLjMsNTEyLC0xMDY3CmwwIC0wCmM0LjcsLTcuMywxMSwtMTEsMTksLTExCkg0MDAwMHY0MEgxMDEyLjMKcy0yNzEuMyw1NjcsLTI3MS4zLDU2N2MtMzguNyw4MC43LC04NCwxNzUsLTEzNiwyODNjLTUyLDEwOCwtODkuMTY3LDE4NS4zLC0xMTEuNSwyMzIKYy0yMi4zLDQ2LjcsLTMzLjgsNzAuMywtMzQuNSw3MWMtNC43LDQuNywtMTIuMyw3LC0yMyw3cy0xMiwtMSwtMTIsLTEKcy0xMDksLTI1MywtMTA5LC0yNTNjLTcyLjcsLTE2OCwtMTA5LjMsLTI1MiwtMTEwLC0yNTJjLTEwLjcsOCwtMjIsMTYuNywtMzQsMjYKYy0yMiwxNy4zLC0zMy4zLDI2LC0zNCwyNnMtMjYsLTI2LC0yNiwtMjZzNzYsLTU5LDc2LC01OXM3NiwtNjAsNzYsLTYwegpNMTAwMSA4MGg0MDAwMDB2NDBoLTQwMDAwMHoiPgo8L3BhdGg+Cjwvc3ZnPg==)​ x − E [ x ] ​ ∗ γ + β

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – quantized tensor
* **weight** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – float tensor that corresponds to the gamma, size C
* **bias** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – float tensor that corresponds to the beta, size C
* **mean** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – float mean value in batch normalization, size C
* **var** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – float tensor for variance, size C
* **eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – a value added to the denominator for numerical stability.
* **output_scale** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – output quantized tensor scale
* **output_zero_point** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – output quantized tensor zero_point

Returns
:   A quantized tensor with batch normalization applied.

Return type
:   [Tensor](../tensors.html#torch.Tensor "torch.Tensor")

Example: 

```
>>> qx = torch.quantize_per_tensor(torch.rand(2, 2, 2, 2), 1.5, 3, torch.quint8)
>>> torch.quantized_batch_norm(qx, torch.ones(2), torch.zeros(2), torch.rand(2), torch.rand(2), 0.00001, 0.2, 2)
tensor([[[[-0.2000, -0.2000],
      [ 1.6000, -0.2000]],

     [[-0.4000, -0.4000],
      [-0.4000,  0.6000]]],

    [[[-0.2000, -0.2000],
      [-0.2000, -0.2000]],

     [[ 0.6000, -0.4000],
      [ 0.6000, -0.4000]]]], size=(2, 2, 2, 2), dtype=torch.quint8,
   quantization_scheme=torch.per_tensor_affine, scale=0.2, zero_point=2)

```

