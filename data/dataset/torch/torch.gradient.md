torch.gradient 
================================================================

torch. gradient ( *input*  , *** , *spacing = 1*  , *dim = None*  , *edge_order = 1* ) → List of Tensors 
:   Estimates the gradient of a function <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            g
           </mi>
<mo>
            :
           </mo>
<msup>
<mi mathvariant="double-struck">
             R
            </mi>
<mi>
             n
            </mi>
</msup>
<mo>
            →
           </mo>
<mi mathvariant="double-struck">
            R
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           g : mathbb{R}^n rightarrow mathbb{R}
          </annotation>
</semantics>
</math> -->g : R n → R g : mathbb{R}^n rightarrow mathbb{R}g : R n → R  in
one or more dimensions using the [second-order accurate central differences method](https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf)  and
either first or second order estimates at the boundaries. 

The gradient of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            g
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           g
          </annotation>
</semantics>
</math> -->g gg  is estimated using samples. By default, when `spacing`  is not
specified, the samples are entirely described by `input`  , and the mapping of input coordinates
to an output is the same as the tensor’s mapping of indices to values. For example, for a three-dimensional `input`  the function described is <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            g
           </mi>
<mo>
            :
           </mo>
<msup>
<mi mathvariant="double-struck">
             R
            </mi>
<mn>
             3
            </mn>
</msup>
<mo>
            →
           </mo>
<mi mathvariant="double-struck">
            R
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           g : mathbb{R}^3 rightarrow mathbb{R}
          </annotation>
</semantics>
</math> -->g : R 3 → R g : mathbb{R}^3 rightarrow mathbb{R}g : R 3 → R  , and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            g
           </mi>
<mo stretchy="false">
            (
           </mo>
<mn>
            1
           </mn>
<mo separator="true">
            ,
           </mo>
<mn>
            2
           </mn>
<mo separator="true">
            ,
           </mo>
<mn>
            3
           </mn>
<mo stretchy="false">
            )
           </mo>
<mtext>
</mtext>
<mo>
            =
           </mo>
<mo>
            =
           </mo>
<mi>
            i
           </mi>
<mi>
            n
           </mi>
<mi>
            p
           </mi>
<mi>
            u
           </mi>
<mi>
            t
           </mi>
<mo stretchy="false">
            [
           </mo>
<mn>
            1
           </mn>
<mo separator="true">
            ,
           </mo>
<mn>
            2
           </mn>
<mo separator="true">
            ,
           </mo>
<mn>
            3
           </mn>
<mo stretchy="false">
            ]
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           g(1, 2, 3) == input[1, 2, 3]
          </annotation>
</semantics>
</math> -->g ( 1 , 2 , 3 ) = = i n p u t [ 1 , 2 , 3 ] g(1, 2, 3) == input[1, 2, 3]g ( 1 , 2 , 3 ) == in p u t [ 1 , 2 , 3 ]  . 

When `spacing`  is specified, it modifies the relationship between `input`  and input coordinates.
This is detailed in the “Keyword Arguments” section below. 

The gradient is estimated by estimating each partial derivative of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            g
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           g
          </annotation>
</semantics>
</math> -->g gg  independently. This estimation is
accurate if <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            g
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           g
          </annotation>
</semantics>
</math> -->g gg  is in <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
             C
            </mi>
<mn>
             3
            </mn>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           C^3
          </annotation>
</semantics>
</math> -->C 3 C^3C 3  (it has at least 3 continuous derivatives), and the estimation can be
improved by providing closer samples. Mathematically, the value at each interior point of a partial derivative
is estimated using [Taylor’s theorem with remainder](https://en.wikipedia.org/wiki/Taylor%27s_theorem)  .
Letting <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            x
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           x
          </annotation>
</semantics>
</math> -->x xx  be an interior point with <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            x
           </mi>
<mo>
            −
           </mo>
<msub>
<mi>
             h
            </mi>
<mi>
             l
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           x-h_l
          </annotation>
</semantics>
</math> -->x − h l x-h_lx − h l ​  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            x
           </mi>
<mo>
            +
           </mo>
<msub>
<mi>
             h
            </mi>
<mi>
             r
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           x+h_r
          </annotation>
</semantics>
</math> -->x + h r x+h_rx + h r ​  be points neighboring
it to the left and right respectively, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            f
           </mi>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mo>
            +
           </mo>
<msub>
<mi>
             h
            </mi>
<mi>
             r
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           f(x+h_r)
          </annotation>
</semantics>
</math> -->f ( x + h r ) f(x+h_r)f ( x + h r ​ )  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            f
           </mi>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mo>
            −
           </mo>
<msub>
<mi>
             h
            </mi>
<mi>
             l
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           f(x-h_l)
          </annotation>
</semantics>
</math> -->f ( x − h l ) f(x-h_l)f ( x − h l ​ )  can be estimated using: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right" columnspacing="" rowspacing="0.25em">
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mi>
                f
               </mi>
<mo stretchy="false">
                (
               </mo>
<mi>
                x
               </mi>
<mo>
                +
               </mo>
<msub>
<mi>
                 h
                </mi>
<mi>
                 r
                </mi>
</msub>
<mo stretchy="false">
                )
               </mo>
<mo>
                =
               </mo>
<mi>
                f
               </mi>
<mo stretchy="false">
                (
               </mo>
<mi>
                x
               </mi>
<mo stretchy="false">
                )
               </mo>
<mo>
                +
               </mo>
<msub>
<mi>
                 h
                </mi>
<mi>
                 r
                </mi>
</msub>
<msup>
<mi>
                 f
                </mi>
<mo lspace="0em" mathvariant="normal" rspace="0em">
                 ′
                </mo>
</msup>
<mo stretchy="false">
                (
               </mo>
<mi>
                x
               </mi>
<mo stretchy="false">
                )
               </mo>
<mo>
                +
               </mo>
<msup>
<msub>
<mi>
                  h
                 </mi>
<mi>
                  r
                 </mi>
</msub>
<mn>
                 2
                </mn>
</msup>
<mfrac>
<mrow>
<msup>
<mi>
                   f
                  </mi>
<mrow>
<mo mathvariant="normal">
                    ′
                   </mo>
<mo mathvariant="normal">
                    ′
                   </mo>
</mrow>
</msup>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  x
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<mn>
                 2
                </mn>
</mfrac>
<mo>
                +
               </mo>
<msup>
<msub>
<mi>
                  h
                 </mi>
<mi>
                  r
                 </mi>
</msub>
<mn>
                 3
                </mn>
</msup>
<mfrac>
<mrow>
<msup>
<mi>
                   f
                  </mi>
<mrow>
<mo mathvariant="normal">
                    ′
                   </mo>
<mo mathvariant="normal">
                    ′
                   </mo>
<mo mathvariant="normal">
                    ′
                   </mo>
</mrow>
</msup>
<mo stretchy="false">
                  (
                 </mo>
<msub>
<mi>
                   ξ
                  </mi>
<mn>
                   1
                  </mn>
</msub>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<mn>
                 6
                </mn>
</mfrac>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 ξ
                </mi>
<mn>
                 1
                </mn>
</msub>
<mo>
                ∈
               </mo>
<mo stretchy="false">
                (
               </mo>
<mi>
                x
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                x
               </mi>
<mo>
                +
               </mo>
<msub>
<mi>
                 h
                </mi>
<mi>
                 r
                </mi>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mi>
                f
               </mi>
<mo stretchy="false">
                (
               </mo>
<mi>
                x
               </mi>
<mo>
                −
               </mo>
<msub>
<mi>
                 h
                </mi>
<mi>
                 l
                </mi>
</msub>
<mo stretchy="false">
                )
               </mo>
<mo>
                =
               </mo>
<mi>
                f
               </mi>
<mo stretchy="false">
                (
               </mo>
<mi>
                x
               </mi>
<mo stretchy="false">
                )
               </mo>
<mo>
                −
               </mo>
<msub>
<mi>
                 h
                </mi>
<mi>
                 l
                </mi>
</msub>
<msup>
<mi>
                 f
                </mi>
<mo lspace="0em" mathvariant="normal" rspace="0em">
                 ′
                </mo>
</msup>
<mo stretchy="false">
                (
               </mo>
<mi>
                x
               </mi>
<mo stretchy="false">
                )
               </mo>
<mo>
                +
               </mo>
<msup>
<msub>
<mi>
                  h
                 </mi>
<mi>
                  l
                 </mi>
</msub>
<mn>
                 2
                </mn>
</msup>
<mfrac>
<mrow>
<msup>
<mi>
                   f
                  </mi>
<mrow>
<mo mathvariant="normal">
                    ′
                   </mo>
<mo mathvariant="normal">
                    ′
                   </mo>
</mrow>
</msup>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  x
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<mn>
                 2
                </mn>
</mfrac>
<mo>
                −
               </mo>
<msup>
<msub>
<mi>
                  h
                 </mi>
<mi>
                  l
                 </mi>
</msub>
<mn>
                 3
                </mn>
</msup>
<mfrac>
<mrow>
<msup>
<mi>
                   f
                  </mi>
<mrow>
<mo mathvariant="normal">
                    ′
                   </mo>
<mo mathvariant="normal">
                    ′
                   </mo>
<mo mathvariant="normal">
                    ′
                   </mo>
</mrow>
</msup>
<mo stretchy="false">
                  (
                 </mo>
<msub>
<mi>
                   ξ
                  </mi>
<mn>
                   2
                  </mn>
</msub>
<mo stretchy="false">
                  )
                 </mo>
</mrow>
<mn>
                 6
                </mn>
</mfrac>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 ξ
                </mi>
<mn>
                 2
                </mn>
</msub>
<mo>
                ∈
               </mo>
<mo stretchy="false">
                (
               </mo>
<mi>
                x
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                x
               </mi>
<mo>
                −
               </mo>
<msub>
<mi>
                 h
                </mi>
<mi>
                 l
                </mi>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
           begin{aligned}
    f(x+h_r) = f(x) + h_r f'(x) + {h_r}^2  frac{f''(x)}{2} + {h_r}^3 frac{f'''(xi_1)}{6}, xi_1 in (x, x+h_r) 
    f(x-h_l) = f(x) - h_l f'(x) + {h_l}^2  frac{f''(x)}{2} - {h_l}^3 frac{f'''(xi_2)}{6}, xi_2 in (x, x-h_l) 
end{aligned}
          </annotation>
</semantics>
</math> -->
f ( x + h r ) = f ( x ) + h r f ′ ( x ) + h r 2 f ′ ′ ( x ) 2 + h r 3 f ′ ′ ′ ( ξ 1 ) 6 , ξ 1 ∈ ( x , x + h r ) f ( x − h l ) = f ( x ) − h l f ′ ( x ) + h l 2 f ′ ′ ( x ) 2 − h l 3 f ′ ′ ′ ( ξ 2 ) 6 , ξ 2 ∈ ( x , x − h l ) begin{aligned}
 f(x+h_r) = f(x) + h_r f'(x) + {h_r}^2 frac{f''(x)}{2} + {h_r}^3 frac{f'''(xi_1)}{6}, xi_1 in (x, x+h_r) 
 f(x-h_l) = f(x) - h_l f'(x) + {h_l}^2 frac{f''(x)}{2} - {h_l}^3 frac{f'''(xi_2)}{6}, xi_2 in (x, x-h_l) 
end{aligned}

f ( x + h r ​ ) = f ( x ) + h r ​ f ′ ( x ) + h r ​ 2 2 f ′′ ( x ) ​ + h r ​ 3 6 f ′′′ ( ξ 1 ​ ) ​ , ξ 1 ​ ∈ ( x , x + h r ​ ) f ( x − h l ​ ) = f ( x ) − h l ​ f ′ ( x ) + h l ​ 2 2 f ′′ ( x ) ​ − h l ​ 3 6 f ′′′ ( ξ 2 ​ ) ​ , ξ 2 ​ ∈ ( x , x − h l ​ ) ​

Using the fact that <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            f
           </mi>
<mo>
            ∈
           </mo>
<msup>
<mi>
             C
            </mi>
<mn>
             3
            </mn>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           f in C^3
          </annotation>
</semantics>
</math> -->f ∈ C 3 f in C^3f ∈ C 3  and solving the linear system, we derive: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mi>
             f
            </mi>
<mo lspace="0em" mathvariant="normal" rspace="0em">
             ′
            </mo>
</msup>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            ≈
           </mo>
<mfrac>
<mrow>
<msup>
<msub>
<mi>
                h
               </mi>
<mi>
                l
               </mi>
</msub>
<mn>
               2
              </mn>
</msup>
<mi>
              f
             </mi>
<mo stretchy="false">
              (
             </mo>
<mi>
              x
             </mi>
<mo>
              +
             </mo>
<msub>
<mi>
               h
              </mi>
<mi>
               r
              </mi>
</msub>
<mo stretchy="false">
              )
             </mo>
<mo>
              −
             </mo>
<msup>
<msub>
<mi>
                h
               </mi>
<mi>
                r
               </mi>
</msub>
<mn>
               2
              </mn>
</msup>
<mi>
              f
             </mi>
<mo stretchy="false">
              (
             </mo>
<mi>
              x
             </mi>
<mo>
              −
             </mo>
<msub>
<mi>
               h
              </mi>
<mi>
               l
              </mi>
</msub>
<mo stretchy="false">
              )
             </mo>
<mo>
              +
             </mo>
<mo stretchy="false">
              (
             </mo>
<msup>
<msub>
<mi>
                h
               </mi>
<mi>
                r
               </mi>
</msub>
<mn>
               2
              </mn>
</msup>
<mo>
              −
             </mo>
<msup>
<msub>
<mi>
                h
               </mi>
<mi>
                l
               </mi>
</msub>
<mn>
               2
              </mn>
</msup>
<mo stretchy="false">
              )
             </mo>
<mi>
              f
             </mi>
<mo stretchy="false">
              (
             </mo>
<mi>
              x
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
<mrow>
<msub>
<mi>
               h
              </mi>
<mi>
               r
              </mi>
</msub>
<msup>
<msub>
<mi>
                h
               </mi>
<mi>
                l
               </mi>
</msub>
<mn>
               2
              </mn>
</msup>
<mo>
              +
             </mo>
<msup>
<msub>
<mi>
                h
               </mi>
<mi>
                r
               </mi>
</msub>
<mn>
               2
              </mn>
</msup>
<msub>
<mi>
               h
              </mi>
<mi>
               l
              </mi>
</msub>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           f'(x) approx frac{ {h_l}^2 f(x+h_r) - {h_r}^2 f(x-h_l)
      + ({h_r}^2-{h_l}^2 ) f(x) }{ {h_r} {h_l}^2 + {h_r}^2 {h_l} }
          </annotation>
</semantics>
</math> -->
f ′ ( x ) ≈ h l 2 f ( x + h r ) − h r 2 f ( x − h l ) + ( h r 2 − h l 2 ) f ( x ) h r h l 2 + h r 2 h l f'(x) approx frac{ {h_l}^2 f(x+h_r) - {h_r}^2 f(x-h_l)
 + ({h_r}^2-{h_l}^2 ) f(x) }{ {h_r} {h_l}^2 + {h_r}^2 {h_l} }

f ′ ( x ) ≈ h r ​ h l ​ 2 + h r ​ 2 h l ​ h l ​ 2 f ( x + h r ​ ) − h r ​ 2 f ( x − h l ​ ) + ( h r ​ 2 − h l ​ 2 ) f ( x ) ​

Note 

We estimate the gradient of functions in complex domain <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             g
            </mi>
<mo>
             :
            </mo>
<msup>
<mi mathvariant="double-struck">
              C
             </mi>
<mi>
              n
             </mi>
</msup>
<mo>
             →
            </mo>
<mi mathvariant="double-struck">
             C
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            g : mathbb{C}^n rightarrow mathbb{C}
           </annotation>
</semantics>
</math> -->g : C n → C g : mathbb{C}^n rightarrow mathbb{C}g : C n → C  in the same way.

The value of each partial derivative at the boundary points is computed differently. See edge_order below. 

Parameters
: **input** ( `Tensor`  ) – the tensor that represents the values of the function

Keyword Arguments
:   * **spacing** ( `scalar`  , `list of scalar`  , `list of Tensor`  , optional) – `spacing`  can be used to modify
how the `input`  tensor’s indices relate to sample coordinates. If `spacing`  is a scalar then
the indices are multiplied by the scalar to produce the coordinates. For example, if `spacing=2`  the
indices (1, 2, 3) become coordinates (2, 4, 6). If `spacing`  is a list of scalars then the corresponding
indices are multiplied. For example, if `spacing=(2, -1, 3)`  the indices (1, 2, 3) become coordinates (2, -2, 9).
Finally, if `spacing`  is a list of one-dimensional tensors then each tensor specifies the coordinates for
the corresponding dimension. For example, if the indices are (1, 2, 3) and the tensors are (t0, t1, t2), then
the coordinates are (t0[1], t1[2], t2[3])
* **dim** ( `int`  , `list of int`  , optional) – the dimension or dimensions to approximate the gradient over. By default
the partial gradient in every dimension is computed. Note that when `dim`  is specified the elements of
the `spacing`  argument must correspond with the specified dims.”
* **edge_order** ( `int`  , optional) – 1 or 2, for [first-order](https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf)  or [second-order](https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf)  estimation of the boundary (“edge”) values, respectively.

Examples: 

```
>>> # Estimates the gradient of f(x)=x^2 at points [-2, -1, 2, 4]
>>> coordinates = (torch.tensor([-2., -1., 1., 4.]),)
>>> values = torch.tensor([4., 1., 1., 16.], )
>>> torch.gradient(values, spacing = coordinates)
(tensor([-3., -2., 2., 5.]),)

>>> # Estimates the gradient of the R^2 -> R function whose samples are
>>> # described by the tensor t. Implicit coordinates are [0, 1] for the outermost
>>> # dimension and [0, 1, 2, 3] for the innermost dimension, and function estimates
>>> # partial derivative for both dimensions.
>>> t = torch.tensor([[1, 2, 4, 8], [10, 20, 40, 80]])
>>> torch.gradient(t)
(tensor([[ 9., 18., 36., 72.],
         [ 9., 18., 36., 72.]]),
 tensor([[ 1.0000, 1.5000, 3.0000, 4.0000],
         [10.0000, 15.0000, 30.0000, 40.0000]]))

>>> # A scalar value for spacing modifies the relationship between tensor indices
>>> # and input coordinates by multiplying the indices to find the
>>> # coordinates. For example, below the indices of the innermost
>>> # 0, 1, 2, 3 translate to coordinates of [0, 2, 4, 6], and the indices of
>>> # the outermost dimension 0, 1 translate to coordinates of [0, 2].
>>> torch.gradient(t, spacing = 2.0) # dim = None (implicitly [0, 1])
(tensor([[ 4.5000, 9.0000, 18.0000, 36.0000],
          [ 4.5000, 9.0000, 18.0000, 36.0000]]),
 tensor([[ 0.5000, 0.7500, 1.5000, 2.0000],
          [ 5.0000, 7.5000, 15.0000, 20.0000]]))
>>> # doubling the spacing between samples halves the estimated partial gradients.

>>>
>>> # Estimates only the partial derivative for dimension 1
>>> torch.gradient(t, dim = 1) # spacing = None (implicitly 1.)
(tensor([[ 1.0000, 1.5000, 3.0000, 4.0000],
         [10.0000, 15.0000, 30.0000, 40.0000]]),)

>>> # When spacing is a list of scalars, the relationship between the tensor
>>> # indices and input coordinates changes based on dimension.
>>> # For example, below, the indices of the innermost dimension 0, 1, 2, 3 translate
>>> # to coordinates of [0, 3, 6, 9], and the indices of the outermost dimension
>>> # 0, 1 translate to coordinates of [0, 2].
>>> torch.gradient(t, spacing = [3., 2.])
(tensor([[ 4.5000, 9.0000, 18.0000, 36.0000],
         [ 4.5000, 9.0000, 18.0000, 36.0000]]),
 tensor([[ 0.3333, 0.5000, 1.0000, 1.3333],
         [ 3.3333, 5.0000, 10.0000, 13.3333]]))

>>> # The following example is a replication of the previous one with explicit
>>> # coordinates.
>>> coords = (torch.tensor([0, 2]), torch.tensor([0, 3, 6, 9]))
>>> torch.gradient(t, spacing = coords)
(tensor([[ 4.5000, 9.0000, 18.0000, 36.0000],
         [ 4.5000, 9.0000, 18.0000, 36.0000]]),
 tensor([[ 0.3333, 0.5000, 1.0000, 1.3333],
         [ 3.3333, 5.0000, 10.0000, 13.3333]]))

```

