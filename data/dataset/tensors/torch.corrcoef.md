torch.corrcoef 
================================================================

torch. corrcoef ( *input* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Estimates the Pearson product-moment correlation coefficient matrix of the variables given by the `input`  matrix,
where rows are the variables and columns are the observations. 

Note 

The correlation coefficient matrix R is computed using the covariance matrix C as given by <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              R
             </mi>
<mrow>
<mi>
               i
              </mi>
<mi>
               j
              </mi>
</mrow>
</msub>
<mo>
             =
            </mo>
<mfrac>
<msub>
<mi>
               C
              </mi>
<mrow>
<mi>
                i
               </mi>
<mi>
                j
               </mi>
</mrow>
</msub>
<msqrt>
<mrow>
<msub>
<mi>
                 C
                </mi>
<mrow>
<mi>
                  i
                 </mi>
<mi>
                  i
                 </mi>
</mrow>
</msub>
<mo>
                ∗
               </mo>
<msub>
<mi>
                 C
                </mi>
<mrow>
<mi>
                  j
                 </mi>
<mi>
                  j
                 </mi>
</mrow>
</msub>
</mrow>
</msqrt>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
            R_{ij} = frac{ C_{ij} } { sqrt{ C_{ii} * C_{jj} } }
           </annotation>
</semantics>
</math> -->R i j = C i j C i i ∗ C j j R_{ij} = frac{ C_{ij} } { sqrt{ C_{ii} * C_{jj} } }R ij ​ = C ii ​ ∗ C jj ​ ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuNTQyOWVtIiBwcmVzZXJ2ZWFzcGVjdHJhdGlvPSJ4TWluWU1pbiBzbGljZSIgdmlld2JveD0iMCAwIDQwMDAwMCAxMDgwIiB3aWR0aD0iNDAwZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik05NSw3MDIKYy0yLjcsMCwtNy4xNywtMi43LC0xMy41LC04Yy01LjgsLTUuMywtOS41LC0xMCwtOS41LC0xNApjMCwtMiwwLjMsLTMuMywxLC00YzEuMywtMi43LDIzLjgzLC0yMC43LDY3LjUsLTU0CmM0NC4yLC0zMy4zLDY1LjgsLTUwLjMsNjYuNSwtNTFjMS4zLC0xLjMsMywtMiw1LC0yYzQuNywwLDguNywzLjMsMTIsMTAKczE3MywzNzgsMTczLDM3OGMwLjcsMCwzNS4zLC03MSwxMDQsLTIxM2M2OC43LC0xNDIsMTM3LjUsLTI4NSwyMDYuNSwtNDI5CmM2OSwtMTQ0LDEwNC41LC0yMTcuNywxMDYuNSwtMjIxCmwwIC0wCmM1LjMsLTkuMywxMiwtMTQsMjAsLTE0Ckg0MDAwMDB2NDBIODQ1LjI3MjQKcy0yMjUuMjcyLDQ2NywtMjI1LjI3Miw0NjdzLTIzNSw0ODYsLTIzNSw0ODZjLTIuNyw0LjcsLTksNywtMTksNwpjLTYsMCwtMTAsLTEsLTEyLC0zcy0xOTQsLTQyMiwtMTk0LC00MjJzLTY1LDQ3LC02NSw0N3oKTTgzNCA4MGg0MDAwMDB2NDBoLTQwMDAwMHoiPgo8L3BhdGg+Cjwvc3ZnPg==)​ C ij ​ ​

Note 

Due to floating point rounding, the resulting array may not be Hermitian and its diagonal elements may not be 1.
The real and imaginary values are clipped to the interval [-1, 1] in an attempt to improve this situation.

Parameters
: **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – A 2D matrix containing multiple variables and observations, or a
Scalar or 1D vector representing a single variable.

Returns
:   (Tensor) The correlation coefficient matrix of the variables.

See also 

[`torch.cov()`](torch.cov.html#torch.cov "torch.cov")  covariance matrix.

Example: 

```
>>> x = torch.tensor([[0, 1, 2], [2, 1, 0]])
>>> torch.corrcoef(x)
tensor([[ 1., -1.],
        [-1.,  1.]])
>>> x = torch.randn(2, 4)
>>> x
tensor([[-0.2678, -0.0908, -0.3766,  0.2780],
        [-0.5812,  0.1535,  0.2387,  0.2350]])
>>> torch.corrcoef(x)
tensor([[1.0000, 0.3582],
        [0.3582, 1.0000]])
>>> torch.corrcoef(x[0])
tensor(1.)

```

