torch.nn.utils.parametrizations.orthogonal 
========================================================================================================================

torch.nn.utils.parametrizations. orthogonal ( *module*  , *name = 'weight'*  , *orthogonal_map = None*  , *** , *use_trivialization = True* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/utils/parametrizations.py#L191) 
:   Apply an orthogonal or unitary parametrization to a matrix or a batch of matrices. 

Letting <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="double-struck">
            K
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           mathbb{K}
          </annotation>
</semantics>
</math> -->K mathbb{K}K  be <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="double-struck">
            R
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           mathbb{R}
          </annotation>
</semantics>
</math> -->R mathbb{R}R  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="double-struck">
            C
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           mathbb{C}
          </annotation>
</semantics>
</math> -->C mathbb{C}C  , the parametrized
matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            Q
           </mi>
<mo>
            ∈
           </mo>
<msup>
<mi mathvariant="double-struck">
             K
            </mi>
<mrow>
<mi>
              m
             </mi>
<mo>
              ×
             </mo>
<mi>
              n
             </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           Q in mathbb{K}^{m times n}
          </annotation>
</semantics>
</math> -->Q ∈ K m × n Q in mathbb{K}^{m times n}Q ∈ K m × n  is **orthogonal** as 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right left" columnspacing="0em" rowspacing="0.25em">
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<msup>
<mi>
                 Q
                </mi>
<mtext>
                 H
                </mtext>
</msup>
<mi>
                Q
               </mi>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mo>
                =
               </mo>
<msub>
<mi mathvariant="normal">
                 I
                </mi>
<mi>
                 n
                </mi>
</msub>
<mpadded width="0px">
<mrow>
<mspace width="2em">
</mspace>
<mtext>
                  if
                 </mtext>
<mi>
                  m
                 </mi>
<mo>
                  ≥
                 </mo>
<mi>
                  n
                 </mi>
</mrow>
</mpadded>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mi>
                Q
               </mi>
<msup>
<mi>
                 Q
                </mi>
<mtext>
                 H
                </mtext>
</msup>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mo>
                =
               </mo>
<msub>
<mi mathvariant="normal">
                 I
                </mi>
<mi>
                 m
                </mi>
</msub>
<mpadded width="0px">
<mrow>
<mspace width="2em">
</mspace>
<mtext>
                  if
                 </mtext>
<mi>
                  m
                 </mi>
<mo>
                  &lt;
                 </mo>
<mi>
                  n
                 </mi>
</mrow>
</mpadded>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
           begin{align*}
    Q^{text{H}}Q &amp;= mathrm{I}_n mathrlap{qquad text{if }m geq n}
    QQ^{text{H}} &amp;= mathrm{I}_m mathrlap{qquad text{if }m &lt; n}
end{align*}
          </annotation>
</semantics>
</math> -->
Q H Q = I n if m ≥ n Q Q H = I m if m < n begin{align*}
 Q^{text{H}}Q &= mathrm{I}_n mathrlap{qquad text{if }m geq n}
 QQ^{text{H}} &= mathrm{I}_m mathrlap{qquad text{if }m < n}
end{align*}

Q H Q Q Q H ​ = I n ​ if m ≥ n = I m ​ if m < n ​

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
             Q
            </mi>
<mtext>
             H
            </mtext>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           Q^{text{H}}
          </annotation>
</semantics>
</math> -->Q H Q^{text{H}}Q H  is the conjugate transpose when <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            Q
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           Q
          </annotation>
</semantics>
</math> -->Q QQ  is complex
and the transpose when <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            Q
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           Q
          </annotation>
</semantics>
</math> -->Q QQ  is real-valued, and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi mathvariant="normal">
             I
            </mi>
<mi>
             n
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           mathrm{I}_n
          </annotation>
</semantics>
</math> -->I n mathrm{I}_nI n ​  is the *n* -dimensional identity matrix.
In plain words, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            Q
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           Q
          </annotation>
</semantics>
</math> -->Q QQ  will have orthonormal columns whenever <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            m
           </mi>
<mo>
            ≥
           </mo>
<mi>
            n
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           m geq n
          </annotation>
</semantics>
</math> -->m ≥ n m geq nm ≥ n  and orthonormal rows otherwise. 

If the tensor has more than two dimensions, we consider it as a batch of matrices of shape *(…, m, n)* . 

The matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            Q
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           Q
          </annotation>
</semantics>
</math> -->Q QQ  may be parametrized via three different `orthogonal_map`  in terms of the original tensor: 

* `"matrix_exp"`  / `"cayley"`  :
the [`matrix_exp()`](torch.matrix_exp.html#torch.matrix_exp "torch.matrix_exp") <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
              Q
             </mi>
<mo>
              =
             </mo>
<mi>
              exp
             </mi>
<mo>
              ⁡
             </mo>
<mo stretchy="false">
              (
             </mo>
<mi>
              A
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             Q = exp(A)
            </annotation>
</semantics>
</math> -->Q = exp ⁡ ( A ) Q = exp(A)Q = exp ( A )  and the [Cayley map](https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map) <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
              Q
             </mi>
<mo>
              =
             </mo>
<mo stretchy="false">
              (
             </mo>
<msub>
<mi mathvariant="normal">
               I
              </mi>
<mi>
               n
              </mi>
</msub>
<mo>
              +
             </mo>
<mi>
              A
             </mi>
<mi mathvariant="normal">
              /
             </mi>
<mn>
              2
             </mn>
<mo stretchy="false">
              )
             </mo>
<mo stretchy="false">
              (
             </mo>
<msub>
<mi mathvariant="normal">
               I
              </mi>
<mi>
               n
              </mi>
</msub>
<mo>
              −
             </mo>
<mi>
              A
             </mi>
<mi mathvariant="normal">
              /
             </mi>
<mn>
              2
             </mn>
<msup>
<mo stretchy="false">
               )
              </mo>
<mrow>
<mo>
                −
               </mo>
<mn>
                1
               </mn>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
             Q = (mathrm{I}_n + A/2)(mathrm{I}_n - A/2)^{-1}
            </annotation>
</semantics>
</math> -->Q = ( I n + A / 2 ) ( I n − A / 2 ) − 1 Q = (mathrm{I}_n + A/2)(mathrm{I}_n - A/2)^{-1}Q = ( I n ​ + A /2 ) ( I n ​ − A /2 ) − 1  are applied to a skew-symmetric <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
              A
             </mi>
</mrow>
<annotation encoding="application/x-tex">
             A
            </annotation>
</semantics>
</math> -->A AA  to give an orthogonal matrix.

* `"householder"`  : computes a product of Householder reflectors
( [`householder_product()`](torch.linalg.householder_product.html#torch.linalg.householder_product "torch.linalg.householder_product")  ).

`"matrix_exp"`  / `"cayley"`  often make the parametrized weight converge faster than `"householder"`  , but they are slower to compute for very thin or very wide matrices. 

If `use_trivialization=True`  (default), the parametrization implements the “Dynamic Trivialization Framework”,
where an extra matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            B
           </mi>
<mo>
            ∈
           </mo>
<msup>
<mi mathvariant="double-struck">
             K
            </mi>
<mrow>
<mi>
              n
             </mi>
<mo>
              ×
             </mo>
<mi>
              n
             </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           B in mathbb{K}^{n times n}
          </annotation>
</semantics>
</math> -->B ∈ K n × n B in mathbb{K}^{n times n}B ∈ K n × n  is stored under `module.parametrizations.weight[0].base`  . This helps the
convergence of the parametrized layer at the expense of some extra memory use.
See [Trivializations for Gradient-Based Optimization on Manifolds](https://arxiv.org/abs/1909.09501)  . 

Initial value of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            Q
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           Q
          </annotation>
</semantics>
</math> -->Q QQ  :
If the original tensor is not parametrized and `use_trivialization=True`  (default), the initial value
of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            Q
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           Q
          </annotation>
</semantics>
</math> -->Q QQ  is that of the original tensor if it is orthogonal (or unitary in the complex case)
and it is orthogonalized via the QR decomposition otherwise (see [`torch.linalg.qr()`](torch.linalg.qr.html#torch.linalg.qr "torch.linalg.qr")  ).
Same happens when it is not parametrized and `orthogonal_map="householder"`  even when `use_trivialization=False`  .
Otherwise, the initial value is the result of the composition of all the registered
parametrizations applied to the original tensor. 

Note 

This function is implemented using the parametrization functionality
in [`register_parametrization()`](torch.nn.utils.parametrize.register_parametrization.html#torch.nn.utils.parametrize.register_parametrization "torch.nn.utils.parametrize.register_parametrization")  .

Parameters
:   * **module** ( [*nn.Module*](torch.nn.Module.html#torch.nn.Module "torch.nn.Module")  ) – module on which to register the parametrization.
* **name** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – name of the tensor to make orthogonal. Default: `"weight"`  .
* **orthogonal_map** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – One of the following: `"matrix_exp"`  , `"cayley"`  , `"householder"`  .
Default: `"matrix_exp"`  if the matrix is square or complex, `"householder"`  otherwise.
* **use_trivialization** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether to use the dynamic trivialization framework.
Default: `True`  .

Returns
:   The original module with an orthogonal parametrization registered to the specified
weight

Return type
:   [*Module*](torch.nn.Module.html#torch.nn.Module "torch.nn.modules.module.Module")

Example: 

```
>>> orth_linear = orthogonal(nn.Linear(20, 40))
>>> orth_linear
ParametrizedLinear(
in_features=20, out_features=40, bias=True
(parametrizations): ModuleDict(
    (weight): ParametrizationList(
    (0): _Orthogonal()
    )
)
)
>>> Q = orth_linear.weight
>>> torch.dist(Q.T @ Q, torch.eye(20))
tensor(4.9332e-07)

```

