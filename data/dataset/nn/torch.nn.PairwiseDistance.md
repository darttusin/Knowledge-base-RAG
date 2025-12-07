PairwiseDistance 
====================================================================

*class* torch.nn. PairwiseDistance ( *p = 2.0*  , *eps = 1e-06*  , *keepdim = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/distance.py#L10) 
:   Computes the pairwise distance between input vectors, or between columns of input matrices. 

Distances are computed using `p`  -norm, with constant `eps`  added to avoid division by zero
if `p`  is negative, i.e.: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mrow>
<mi mathvariant="normal">
             d
            </mi>
<mi mathvariant="normal">
             i
            </mi>
<mi mathvariant="normal">
             s
            </mi>
<mi mathvariant="normal">
             t
            </mi>
</mrow>
<mrow>
<mo fence="true">
             (
            </mo>
<mi>
             x
            </mi>
<mo separator="true">
             ,
            </mo>
<mi>
             y
            </mi>
<mo fence="true">
             )
            </mo>
</mrow>
<mo>
            =
           </mo>
<msub>
<mrow>
<mo fence="true">
              ∥
             </mo>
<mi>
              x
             </mi>
<mo>
              −
             </mo>
<mi>
              y
             </mi>
<mo>
              +
             </mo>
<mi>
              ϵ
             </mi>
<mi>
              e
             </mi>
<mo fence="true">
              ∥
             </mo>
</mrow>
<mi>
             p
            </mi>
</msub>
<mo separator="true">
            ,
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           mathrm{dist}left(x, yright) = leftVert x-y + epsilon e rightVert_p,
          </annotation>
</semantics>
</math> -->
d i s t ( x , y ) = ∥ x − y + ϵ e ∥ p , mathrm{dist}left(x, yright) = leftVert x-y + epsilon e rightVert_p,

dist ( x , y ) = ∥ x − y + ϵe ∥ p ​ ,

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            e
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           e
          </annotation>
</semantics>
</math> -->e ee  is the vector of ones and the `p`  -norm is given by. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
            ∥
           </mi>
<mi>
            x
           </mi>
<msub>
<mi mathvariant="normal">
             ∥
            </mi>
<mi>
             p
            </mi>
</msub>
<mo>
            =
           </mo>
<msup>
<mrow>
<mo fence="true">
              (
             </mo>
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
<mi mathvariant="normal">
              ∣
             </mi>
<msub>
<mi>
               x
              </mi>
<mi>
               i
              </mi>
</msub>
<msup>
<mi mathvariant="normal">
               ∣
              </mi>
<mi>
               p
              </mi>
</msup>
<mo fence="true">
              )
             </mo>
</mrow>
<mrow>
<mn>
              1
             </mn>
<mi mathvariant="normal">
              /
             </mi>
<mi>
              p
             </mi>
</mrow>
</msup>
<mi mathvariant="normal">
            .
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           Vert x Vert _p = left( sum_{i=1}^n  vert x_i vert ^ p right) ^ {1/p}.
          </annotation>
</semantics>
</math> -->
∥ x ∥ p = ( ∑ i = 1 n ∣ x i ∣ p ) 1 / p . Vert x Vert _p = left( sum_{i=1}^n vert x_i vert ^ p right) ^ {1/p}.

∥ x ∥ p ​ = ( i = 1 ∑ n ​ ∣ x i ​ ∣ p ) 1/ p .

Parameters
:   * **p** ( *real* *,* *optional*  ) – the norm degree. Can be negative. Default: 2
* **eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – Small value to avoid division by zero.
Default: 1e-6
* **keepdim** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Determines whether or not to keep the vector dimension.
Default: False

Shape:
:   * Input1: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                D
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, D)
              </annotation>
</semantics>
</math> -->( N , D ) (N, D)( N , D )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                D
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (D)
              </annotation>
</semantics>
</math> -->( D ) (D)( D )  where *N = batch dimension* and *D = vector dimension*

* Input2: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                D
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, D)
              </annotation>
</semantics>
</math> -->( N , D ) (N, D)( N , D )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                D
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (D)
              </annotation>
</semantics>
</math> -->( D ) (D)( D )  , same shape as the Input1

* Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N)
              </annotation>
</semantics>
</math> -->( N ) (N)( N )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               ()
              </annotation>
</semantics>
</math> -->( ) ()( )  based on input dimension.
If `keepdim`  is `True`  , then <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                N
               </mi>
<mo separator="true">
                ,
               </mo>
<mn>
                1
               </mn>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, 1)
              </annotation>
</semantics>
</math> -->( N , 1 ) (N, 1)( N , 1 )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mn>
                1
               </mn>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (1)
              </annotation>
</semantics>
</math> -->( 1 ) (1)( 1 )  based on input dimension.

Examples 

```
>>> pdist = nn.PairwiseDistance(p=2)
>>> input1 = torch.randn(100, 128)
>>> input2 = torch.randn(100, 128)
>>> output = pdist(input1, input2)

```

