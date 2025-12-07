CosineSimilarity 
====================================================================

*class* torch.nn. CosineSimilarity ( *dim = 1*  , *eps = 1e-08* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/distance.py#L61) 
:   Returns cosine similarity between <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             x
            </mi>
<mn>
             1
            </mn>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           x_1
          </annotation>
</semantics>
</math> -->x 1 x_1x 1 ​  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             x
            </mi>
<mn>
             2
            </mn>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           x_2
          </annotation>
</semantics>
</math> -->x 2 x_2x 2 ​  , computed along *dim* . 

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
<mo>
              ⋅
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
<mi mathvariant="normal">
            .
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           text{similarity} = dfrac{x_1 cdot x_2}{max(Vert x_1 Vert _2 cdot Vert x_2 Vert _2, epsilon)}.
          </annotation>
</semantics>
</math> -->
similarity = x 1 ⋅ x 2 max ⁡ ( ∥ x 1 ∥ 2 ⋅ ∥ x 2 ∥ 2 , ϵ ) . text{similarity} = dfrac{x_1 cdot x_2}{max(Vert x_1 Vert _2 cdot Vert x_2 Vert _2, epsilon)}.

similarity = max ( ∥ x 1 ​ ∥ 2 ​ ⋅ ∥ x 2 ​ ∥ 2 ​ , ϵ ) x 1 ​ ⋅ x 2 ​ ​ .

Parameters
:   * **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – Dimension where cosine similarity is computed. Default: 1
* **eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – Small value to avoid division by zero.
Default: 1e-8

Shape:
:   * Input1: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<msub>
<mo>
                 ∗
                </mo>
<mn>
                 1
                </mn>
</msub>
<mo separator="true">
                ,
               </mo>
<mi>
                D
               </mi>
<mo separator="true">
                ,
               </mo>
<msub>
<mo>
                 ∗
                </mo>
<mn>
                 2
                </mn>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (ast_1, D, ast_2)
              </annotation>
</semantics>
</math> -->( ∗ 1 , D , ∗ 2 ) (ast_1, D, ast_2)( ∗ 1 ​ , D , ∗ 2 ​ )  where D is at position *dim*

* Input2: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<msub>
<mo>
                 ∗
                </mo>
<mn>
                 1
                </mn>
</msub>
<mo separator="true">
                ,
               </mo>
<mi>
                D
               </mi>
<mo separator="true">
                ,
               </mo>
<msub>
<mo>
                 ∗
                </mo>
<mn>
                 2
                </mn>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (ast_1, D, ast_2)
              </annotation>
</semantics>
</math> -->( ∗ 1 , D , ∗ 2 ) (ast_1, D, ast_2)( ∗ 1 ​ , D , ∗ 2 ​ )  , same number of dimensions as x1, matching x1 size at dimension *dim* ,
and broadcastable with x1 at other dimensions.

* Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<msub>
<mo>
                 ∗
                </mo>
<mn>
                 1
                </mn>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mo>
                 ∗
                </mo>
<mn>
                 2
                </mn>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (ast_1, ast_2)
              </annotation>
</semantics>
</math> -->( ∗ 1 , ∗ 2 ) (ast_1, ast_2)( ∗ 1 ​ , ∗ 2 ​ )

Examples 

```
>>> input1 = torch.randn(100, 128)
>>> input2 = torch.randn(100, 128)
>>> cos = nn.CosineSimilarity(dim=1, eps=1e-6)
>>> output = cos(input1, input2)

```

