torch.nn.functional.normalize 
==============================================================================================

torch.nn.functional. normalize ( *input*  , *p = 2.0*  , *dim = 1*  , *eps = 1e-12*  , *out = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L5537) 
:   Perform <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             L
            </mi>
<mi>
             p
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           L_p
          </annotation>
</semantics>
</math> -->L p L_pL p ​  normalization of inputs over specified dimension. 

For a tensor `input`  of sizes <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             n
            </mi>
<mn>
             0
            </mn>
</msub>
<mo separator="true">
            ,
           </mo>
<mi mathvariant="normal">
            .
           </mi>
<mi mathvariant="normal">
            .
           </mi>
<mi mathvariant="normal">
            .
           </mi>
<mo separator="true">
            ,
           </mo>
<msub>
<mi>
             n
            </mi>
<mrow>
<mi>
              d
             </mi>
<mi>
              i
             </mi>
<mi>
              m
             </mi>
</mrow>
</msub>
<mo separator="true">
            ,
           </mo>
<mi mathvariant="normal">
            .
           </mi>
<mi mathvariant="normal">
            .
           </mi>
<mi mathvariant="normal">
            .
           </mi>
<mo separator="true">
            ,
           </mo>
<msub>
<mi>
             n
            </mi>
<mi>
             k
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (n_0, ..., n_{dim}, ..., n_k)
          </annotation>
</semantics>
</math> -->( n 0 , . . . , n d i m , . . . , n k ) (n_0, ..., n_{dim}, ..., n_k)( n 0 ​ , ... , n d im ​ , ... , n k ​ )  , each <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             n
            </mi>
<mrow>
<mi>
              d
             </mi>
<mi>
              i
             </mi>
<mi>
              m
             </mi>
</mrow>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           n_{dim}
          </annotation>
</semantics>
</math> -->n d i m n_{dim}n d im ​  -element vector <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            v
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           v
          </annotation>
</semantics>
</math> -->v vv  along dimension `dim`  is transformed as 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            v
           </mi>
<mo>
            =
           </mo>
<mfrac>
<mi>
             v
            </mi>
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
<mo stretchy="false">
              ∥
             </mo>
<mi>
              v
             </mi>
<msub>
<mo stretchy="false">
               ∥
              </mo>
<mi>
               p
              </mi>
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
<mi mathvariant="normal">
            .
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           v = frac{v}{max(lVert v rVert_p, epsilon)}.
          </annotation>
</semantics>
</math> -->
v = v max ⁡ ( ∥ v ∥ p , ϵ ) . v = frac{v}{max(lVert v rVert_p, epsilon)}.

v = max (∥ v ∥ p ​ , ϵ ) v ​ .

With the default arguments it uses the Euclidean norm over vectors along dimension <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
            1
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           1
          </annotation>
</semantics>
</math> -->1 11  for normalization. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – input tensor of any shape
* **p** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – the exponent value in the norm formulation. Default: 2
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *of* *ints*  ) – the dimension to reduce. Default: 1
* **eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – small value to avoid division by zero. Default: 1e-12
* **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor. If `out`  is used, this
operation won’t be differentiable.

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

