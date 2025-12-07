torch.nn.functional.pdist 
======================================================================================

torch.nn.functional. pdist ( *input*  , *p = 2* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Computes the p-norm distance between every pair of row vectors in the input.
This is identical to the upper triangular portion, excluding the diagonal, of *torch.norm(input[:, None] - input, dim=2, p=p)* . This function will be faster
if the rows are contiguous. 

If input has shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            N
           </mi>
<mo>
            ×
           </mo>
<mi>
            M
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           N times M
          </annotation>
</semantics>
</math> -->N × M N times MN × M  then the output will have shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mfrac>
<mn>
             1
            </mn>
<mn>
             2
            </mn>
</mfrac>
<mi>
            N
           </mi>
<mo stretchy="false">
            (
           </mo>
<mi>
            N
           </mi>
<mo>
            −
           </mo>
<mn>
            1
           </mn>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           frac{1}{2} N (N - 1)
          </annotation>
</semantics>
</math> -->1 2 N ( N − 1 ) frac{1}{2} N (N - 1)2 1 ​ N ( N − 1 )  . 

This function is equivalent to `scipy.spatial.distance.pdist(input, 'minkowski', p=p)`  if <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->p = 0 p = 0p = 0  it is
equivalent to `scipy.spatial.distance.pdist(input, 'hamming') * M`  .
When <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->p = ∞ p = inftyp = ∞  , the closest scipy function is `scipy.spatial.distance.pdist(xn, lambda x, y: np.abs(x - y).max())`  . 

Parameters
:   * **input** – input tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                N
               </mi>
<mo>
                ×
               </mo>
<mi>
                M
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               N times M
              </annotation>
</semantics>
</math> -->N × M N times MN × M  .

* **p** – p value for the p-norm distance to calculate between each vector pair <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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

