GLU 
==========================================

*class* torch.nn. GLU ( *dim = -1* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L665) 
:   Applies the gated linear unit function. 

<!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mrow>
<mi>
             G
            </mi>
<mi>
             L
            </mi>
<mi>
             U
            </mi>
</mrow>
<mo stretchy="false">
            (
           </mo>
<mi>
            a
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            b
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mi>
            a
           </mi>
<mo>
            ⊗
           </mo>
<mi>
            σ
           </mi>
<mo stretchy="false">
            (
           </mo>
<mi>
            b
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           {GLU}(a, b)= a otimes sigma(b)
          </annotation>
</semantics>
</math> -->G L U ( a , b ) = a ⊗ σ ( b ) {GLU}(a, b)= a otimes sigma(b)G LU ( a , b ) = a ⊗ σ ( b )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            a
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           a
          </annotation>
</semantics>
</math> -->a aa  is the first half
of the input matrices and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            b
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           b
          </annotation>
</semantics>
</math> -->b bb  is the second half. 

Parameters
: **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the dimension on which to split the input. Default: -1

Shape:
:   * Input: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                N
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
               (ast_1, N, ast_2)
              </annotation>
</semantics>
</math> -->( ∗ 1 , N , ∗ 2 ) (ast_1, N, ast_2)( ∗ 1 ​ , N , ∗ 2 ​ )  where *** means, any number of additional
dimensions

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
<mi>
                M
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
               (ast_1, M, ast_2)
              </annotation>
</semantics>
</math> -->( ∗ 1 , M , ∗ 2 ) (ast_1, M, ast_2)( ∗ 1 ​ , M , ∗ 2 ​ )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                M
               </mi>
<mo>
                =
               </mo>
<mi>
                N
               </mi>
<mi mathvariant="normal">
                /
               </mi>
<mn>
                2
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               M=N/2
              </annotation>
</semantics>
</math> -->M = N / 2 M=N/2M = N /2

![../_images/GLU.png](../_images/GLU.png)

Examples: 

```
>>> m = nn.GLU()
>>> input = torch.randn(4, 2)
>>> output = m(input)

```

