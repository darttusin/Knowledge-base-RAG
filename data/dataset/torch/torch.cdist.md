torch.cdist 
==========================================================

torch. cdist ( *x1*  , *x2*  , *p = 2.0*  , *compute_mode = 'use_mm_for_euclid_dist_if_necessary'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/functional.py#L1458) 
:   Computes batched the p-norm distance between each pair of the two collections of row vectors. 

Parameters
:   * **x1** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – input tensor where the last two dimensions represent the points and the feature dimension respectively.
The shape can be <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
                 D
                </mi>
<mn>
                 1
                </mn>
</msub>
<mo>
                ×
               </mo>
<msub>
<mi>
                 D
                </mi>
<mn>
                 2
                </mn>
</msub>
<mo>
                ×
               </mo>
<mo>
                ⋯
               </mo>
<mo>
                ×
               </mo>
<msub>
<mi>
                 D
                </mi>
<mi>
                 n
                </mi>
</msub>
<mo>
                ×
               </mo>
<mi>
                P
               </mi>
<mo>
                ×
               </mo>
<mi>
                M
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               D_1 times D_2 times cdots times D_n times P times M
              </annotation>
</semantics>
</math> -->D 1 × D 2 × ⋯ × D n × P × M D_1 times D_2 times cdots times D_n times P times MD 1 ​ × D 2 ​ × ⋯ × D n ​ × P × M  ,
where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                P
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               P
              </annotation>
</semantics>
</math> -->P PP  is the number of points and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                M
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               M
              </annotation>
</semantics>
</math> -->M MM  is the feature dimension.

* **x2** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – input tensor where the last two dimensions also represent the points and the feature dimension respectively.
The shape can be <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msubsup>
<mi>
                 D
                </mi>
<mn>
                 1
                </mn>
<mo lspace="0em" mathvariant="normal" rspace="0em">
                 ′
                </mo>
</msubsup>
<mo>
                ×
               </mo>
<msubsup>
<mi>
                 D
                </mi>
<mn>
                 2
                </mn>
<mo lspace="0em" mathvariant="normal" rspace="0em">
                 ′
                </mo>
</msubsup>
<mo>
                ×
               </mo>
<mo>
                ⋯
               </mo>
<mo>
                ×
               </mo>
<msubsup>
<mi>
                 D
                </mi>
<mi>
                 m
                </mi>
<mo lspace="0em" mathvariant="normal" rspace="0em">
                 ′
                </mo>
</msubsup>
<mo>
                ×
               </mo>
<mi>
                R
               </mi>
<mo>
                ×
               </mo>
<mi>
                M
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               D_1' times D_2' times cdots times D_m' times R times M
              </annotation>
</semantics>
</math> -->D 1 ′ × D 2 ′ × ⋯ × D m ′ × R × M D_1' times D_2' times cdots times D_m' times R times MD 1 ′ ​ × D 2 ′ ​ × ⋯ × D m ′ ​ × R × M  ,
where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                R
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               R
              </annotation>
</semantics>
</math> -->R RR  is the number of points and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                M
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               M
              </annotation>
</semantics>
</math> -->M MM  is the feature dimension,
which should match the feature dimension of *x1* .

* **p** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – p value for the p-norm distance to calculate between each vector pair <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
                ∈
               </mo>
<mo stretchy="false">
                [
               </mo>
<mn>
                0
               </mn>
<mo separator="true">
                ,
               </mo>
<mi mathvariant="normal">
                ∞
               </mi>
<mo stretchy="false">
                ]
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               in [0, infty]
              </annotation>
</semantics>
</math> -->∈ [ 0 , ∞ ] in [0, infty]∈ [ 0 , ∞ ]  .

* **compute_mode** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – ‘use_mm_for_euclid_dist_if_necessary’ - will use matrix multiplication approach to calculate
euclidean distance (p = 2) if P > 25 or R > 25
‘use_mm_for_euclid_dist’ - will always use matrix multiplication approach to calculate
euclidean distance (p = 2)
‘donot_use_mm_for_euclid_dist’ - will never use matrix multiplication approach to calculate
euclidean distance (p = 2)
Default: use_mm_for_euclid_dist_if_necessary.

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

If x1 has shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            B
           </mi>
<mo>
            ×
           </mo>
<mi>
            P
           </mi>
<mo>
            ×
           </mo>
<mi>
            M
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           B times P times M
          </annotation>
</semantics>
</math> -->B × P × M B times P times MB × P × M  and x2 has shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            B
           </mi>
<mo>
            ×
           </mo>
<mi>
            R
           </mi>
<mo>
            ×
           </mo>
<mi>
            M
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           B times R times M
          </annotation>
</semantics>
</math> -->B × R × M B times R times MB × R × M  then the
output will have shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            B
           </mi>
<mo>
            ×
           </mo>
<mi>
            P
           </mi>
<mo>
            ×
           </mo>
<mi>
            R
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           B times P times R
          </annotation>
</semantics>
</math> -->B × P × R B times P times RB × P × R  . 

This function is equivalent to *scipy.spatial.distance.cdist(input,’minkowski’, p=p)* if <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            p
           </mi>
<mo>
            ∈
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
<mi mathvariant="normal">
            ∞
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           p in (0, infty)
          </annotation>
</semantics>
</math> -->p ∈ ( 0 , ∞ ) p in (0, infty)p ∈ ( 0 , ∞ )  . When <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            p
           </mi>
<mo>
            =
           </mo>
<mn>
            0
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           p = 0
          </annotation>
</semantics>
</math> -->p = 0 p = 0p = 0  it is equivalent to *scipy.spatial.distance.cdist(input, ‘hamming’) * M* . When <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            p
           </mi>
<mo>
            =
           </mo>
<mi mathvariant="normal">
            ∞
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           p = infty
          </annotation>
</semantics>
</math> -->p = ∞ p = inftyp = ∞  , the closest
scipy function is *scipy.spatial.distance.cdist(xn, lambda x, y: np.abs(x - y).max())* . 

Example 

```
>>> a = torch.tensor([[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 1.059]])
>>> a
tensor([[ 0.9041,  0.0196],
        [-0.3108, -2.4423],
        [-0.4821,  1.0590]])
>>> b = torch.tensor([[-2.1763, -0.4713], [-0.6986, 1.3702]])
>>> b
tensor([[-2.1763, -0.4713],
        [-0.6986,  1.3702]])
>>> torch.cdist(a, b, p=2)
tensor([[3.1193, 2.0959],
        [2.7138, 3.8322],
        [2.2830, 0.3791]])

```

