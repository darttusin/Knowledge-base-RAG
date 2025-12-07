torch.cov 
======================================================

torch. cov ( *input*  , *** , *correction = 1*  , *fweights = None*  , *aweights = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Estimates the covariance matrix of the variables given by the `input`  matrix, where rows are
the variables and columns are the observations. 

A covariance matrix is a square matrix giving the covariance of each pair of variables. The diagonal contains
the variance of each variable (covariance of a variable with itself). By definition, if `input`  represents
a single variable (Scalar or 1D) then its variance is returned. 

The sample covariance of the variables <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->x xx  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            y
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           y
          </annotation>
</semantics>
</math> -->y yy  is given by: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            cov
           </mtext>
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
            y
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mfrac>
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
               N
              </mi>
</munderover>
<mo stretchy="false">
              (
             </mo>
<msub>
<mi>
               x
              </mi>
<mi>
               i
              </mi>
</msub>
<mo>
              −
             </mo>
<mover accent="true">
<mi>
               x
              </mi>
<mo>
               ˉ
              </mo>
</mover>
<mo stretchy="false">
              )
             </mo>
<mo stretchy="false">
              (
             </mo>
<msub>
<mi>
               y
              </mi>
<mi>
               i
              </mi>
</msub>
<mo>
              −
             </mo>
<mover accent="true">
<mi>
               y
              </mi>
<mo>
               ˉ
              </mo>
</mover>
<mo stretchy="false">
              )
             </mo>
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
<mn>
              0
             </mn>
<mo separator="true">
              ,
             </mo>
<mtext>
</mtext>
<mi>
              N
             </mi>
<mtext>
</mtext>
<mo>
              −
             </mo>
<mtext>
</mtext>
<mi>
              δ
             </mi>
<mi>
              N
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           text{cov}(x,y) = frac{sum^{N}_{i = 1}(x_{i} - bar{x})(y_{i} - bar{y})}{max(0,~N~-~delta N)}
          </annotation>
</semantics>
</math> -->
cov ( x , y ) = ∑ i = 1 N ( x i − x ˉ ) ( y i − y ˉ ) max ⁡ ( 0 , N − δ N ) text{cov}(x,y) = frac{sum^{N}_{i = 1}(x_{i} - bar{x})(y_{i} - bar{y})}{max(0,~N~-~delta N)}

cov ( x , y ) = max ( 0 , N − δ N ) ∑ i = 1 N ​ ( x i ​ − x ˉ ) ( y i ​ − y ˉ ​ ) ​

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mover accent="true">
<mi>
             x
            </mi>
<mo>
             ˉ
            </mo>
</mover>
</mrow>
<annotation encoding="application/x-tex">
           bar{x}
          </annotation>
</semantics>
</math> -->x ˉ bar{x}x ˉ  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mover accent="true">
<mi>
             y
            </mi>
<mo>
             ˉ
            </mo>
</mover>
</mrow>
<annotation encoding="application/x-tex">
           bar{y}
          </annotation>
</semantics>
</math> -->y ˉ bar{y}y ˉ ​  are the simple means of the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->x xx  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            y
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           y
          </annotation>
</semantics>
</math> -->y yy  respectively, and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            δ
           </mi>
<mi>
            N
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           delta N
          </annotation>
</semantics>
</math> -->δ N delta Nδ N  is the `correction`  . 

If `fweights`  and/or `aweights`  are provided, the weighted covariance
is calculated, which is given by: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mtext>
             cov
            </mtext>
<mi>
             w
            </mi>
</msub>
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
            y
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mfrac>
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
               N
              </mi>
</munderover>
<msub>
<mi>
               w
              </mi>
<mi>
               i
              </mi>
</msub>
<mo stretchy="false">
              (
             </mo>
<msub>
<mi>
               x
              </mi>
<mi>
               i
              </mi>
</msub>
<mo>
              −
             </mo>
<msubsup>
<mi>
               μ
              </mi>
<mi>
               x
              </mi>
<mo>
               ∗
              </mo>
</msubsup>
<mo stretchy="false">
              )
             </mo>
<mo stretchy="false">
              (
             </mo>
<msub>
<mi>
               y
              </mi>
<mi>
               i
              </mi>
</msub>
<mo>
              −
             </mo>
<msubsup>
<mi>
               μ
              </mi>
<mi>
               y
              </mi>
<mo>
               ∗
              </mo>
</msubsup>
<mo stretchy="false">
              )
             </mo>
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
<mn>
              0
             </mn>
<mo separator="true">
              ,
             </mo>
<mtext>
</mtext>
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
               N
              </mi>
</munderover>
<msub>
<mi>
               w
              </mi>
<mi>
               i
              </mi>
</msub>
<mtext>
</mtext>
<mo>
              −
             </mo>
<mtext>
</mtext>
<mfrac>
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
                 N
                </mi>
</munderover>
<msub>
<mi>
                 w
                </mi>
<mi>
                 i
                </mi>
</msub>
<msub>
<mi>
                 a
                </mi>
<mi>
                 i
                </mi>
</msub>
</mrow>
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
                 N
                </mi>
</munderover>
<msub>
<mi>
                 w
                </mi>
<mi>
                 i
                </mi>
</msub>
</mrow>
</mfrac>
<mtext>
</mtext>
<mi>
              δ
             </mi>
<mi>
              N
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           text{cov}_w(x,y) = frac{sum^{N}_{i = 1}w_i(x_{i} - mu_x^*)(y_{i} - mu_y^*)}
{max(0,~sum^{N}_{i = 1}w_i~-~frac{sum^{N}_{i = 1}w_ia_i}{sum^{N}_{i = 1}w_i}~delta N)}
          </annotation>
</semantics>
</math> -->
cov w ( x , y ) = ∑ i = 1 N w i ( x i − μ x ∗ ) ( y i − μ y ∗ ) max ⁡ ( 0 , ∑ i = 1 N w i − ∑ i = 1 N w i a i ∑ i = 1 N w i δ N ) text{cov}_w(x,y) = frac{sum^{N}_{i = 1}w_i(x_{i} - mu_x^*)(y_{i} - mu_y^*)}
{max(0,~sum^{N}_{i = 1}w_i~-~frac{sum^{N}_{i = 1}w_ia_i}{sum^{N}_{i = 1}w_i}~delta N)}

cov w ​ ( x , y ) = max ( 0 , ∑ i = 1 N ​ w i ​ − ∑ i = 1 N ​ w i ​ ∑ i = 1 N ​ w i ​ a i ​ ​ δ N ) ∑ i = 1 N ​ w i ​ ( x i ​ − μ x ∗ ​ ) ( y i ​ − μ y ∗ ​ ) ​

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            w
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           w
          </annotation>
</semantics>
</math> -->w ww  denotes `fweights`  or `aweights`  ( `f`  and `a`  for brevity) based on whichever is
provided, or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            w
           </mi>
<mo>
            =
           </mo>
<mi>
            f
           </mi>
<mo>
            ×
           </mo>
<mi>
            a
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           w = f times a
          </annotation>
</semantics>
</math> -->w = f × a w = f times aw = f × a  if both are provided, and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msubsup>
<mi>
             μ
            </mi>
<mi>
             x
            </mi>
<mo>
             ∗
            </mo>
</msubsup>
<mo>
            =
           </mo>
<mfrac>
<mrow>
<msubsup>
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
               N
              </mi>
</msubsup>
<msub>
<mi>
               w
              </mi>
<mi>
               i
              </mi>
</msub>
<msub>
<mi>
               x
              </mi>
<mi>
               i
              </mi>
</msub>
</mrow>
<mrow>
<msubsup>
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
               N
              </mi>
</msubsup>
<msub>
<mi>
               w
              </mi>
<mi>
               i
              </mi>
</msub>
</mrow>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
           mu_x^* = frac{sum^{N}_{i = 1}w_ix_{i} }{sum^{N}_{i = 1}w_i}
          </annotation>
</semantics>
</math> -->μ x ∗ = ∑ i = 1 N w i x i ∑ i = 1 N w i mu_x^* = frac{sum^{N}_{i = 1}w_ix_{i} }{sum^{N}_{i = 1}w_i}μ x ∗ ​ = ∑ i = 1 N ​ w i ​ ∑ i = 1 N ​ w i ​ x i ​ ​  is the weighted mean of the variable. If not
provided, `f`  and/or `a`  can be seen as a <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn mathvariant="double-struck">
            1
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           mathbb{1}
          </annotation>
</semantics>
</math> -->1 mathbb{1}1  vector of appropriate size. 

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – A 2D matrix containing multiple variables and observations, or a
Scalar or 1D vector representing a single variable.

Keyword Arguments
:   * **correction** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – difference between the sample size and sample degrees of freedom.
Defaults to Bessel’s correction, `correction = 1`  which returns the unbiased estimate,
even if both `fweights`  and `aweights`  are specified. `correction = 0`  will return the simple average. Defaults to `1`  .
* **fweights** ( [*tensor*](torch.tensor.html#torch.tensor "torch.tensor") *,* *optional*  ) – A Scalar or 1D tensor of observation vector frequencies representing the number of
times each observation should be repeated. Its numel must equal the number of columns of `input`  .
Must have integral dtype. Ignored if `None`  . Defaults to `None`  .
* **aweights** ( [*tensor*](torch.tensor.html#torch.tensor "torch.tensor") *,* *optional*  ) – A Scalar or 1D array of observation vector weights.
These relative weights are typically large for observations considered “important” and smaller for
observations considered less “important”. Its numel must equal the number of columns of `input`  .
Must have floating point dtype. Ignored if `None`  . Defaults to `None`  .

Returns
:   (Tensor) The covariance matrix of the variables.

See also 

[`torch.corrcoef()`](torch.corrcoef.html#torch.corrcoef "torch.corrcoef")  normalized covariance matrix.

Example: 

```
>>> x = torch.tensor([[0, 2], [1, 1], [2, 0]]).T
>>> x
tensor([[0, 1, 2],
        [2, 1, 0]])
>>> torch.cov(x)
tensor([[ 1., -1.],
        [-1.,  1.]])
>>> torch.cov(x, correction=0)
tensor([[ 0.6667, -0.6667],
        [-0.6667,  0.6667]])
>>> fw = torch.randint(1, 10, (3,))
>>> fw
tensor([1, 6, 9])
>>> aw = torch.rand(3)
>>> aw
tensor([0.4282, 0.0255, 0.4144])
>>> torch.cov(x, fweights=fw, aweights=aw)
tensor([[ 0.4169, -0.4169],
        [-0.4169,  0.4169]])

```

