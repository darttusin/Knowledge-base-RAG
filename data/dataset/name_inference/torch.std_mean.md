torch.std_mean 
=================================================================

torch. std_mean ( *input*  , *dim = None*  , *** , *correction = 1*  , *keepdim = False*  , *out = None* ) 
:   Calculates the standard deviation and mean over the dimensions specified by `dim`  . `dim`  can be a single dimension, list of dimensions, or `None`  to reduce over all dimensions. 

The standard deviation ( <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            σ
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           sigma
          </annotation>
</semantics>
</math> -->σ sigmaσ  ) is calculated as 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            σ
           </mi>
<mo>
            =
           </mo>
<msqrt>
<mrow>
<mfrac>
<mn>
               1
              </mn>
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
<mo>
                −
               </mo>
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
                0
               </mn>
</mrow>
<mrow>
<mi>
                N
               </mi>
<mo>
                −
               </mo>
<mn>
                1
               </mn>
</mrow>
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
<msup>
<mo stretchy="false">
               )
              </mo>
<mn>
               2
              </mn>
</msup>
</mrow>
</msqrt>
</mrow>
<annotation encoding="application/x-tex">
           sigma = sqrt{frac{1}{max(0,~N - delta N)}sum_{i=0}^{N-1}(x_i-bar{x})^2}
          </annotation>
</semantics>
</math> -->
σ = 1 max ⁡ ( 0 , N − δ N ) ∑ i = 0 N − 1 ( x i − x ˉ ) 2 sigma = sqrt{frac{1}{max(0,~N - delta N)}sum_{i=0}^{N-1}(x_i-bar{x})^2}

σ = max ( 0 , N − δ N ) 1 ​ i = 0 ∑ N − 1 ​ ( x i ​ − x ˉ ) 2 ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjMuMzczOGVtIiBwcmVzZXJ2ZWFzcGVjdHJhdGlvPSJ4TWluWU1pbiBzbGljZSIgdmlld2JveD0iMCAwIDQwMDAwMCAzMzczIiB3aWR0aD0iNDAwZW0iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik03MDIgODBINDAwMDAwNDAKSDc0MnYzMjM5bC00IDQtNCA0Yy0uNjY3LjcgLTIgMS41LTQgMi41cy00LjE2NyAxLjgzMy02LjUgMi41LTUuNSAxLTkuNSAxCmgtMTJsLTI4LTg0Yy0xNi42NjctNTItOTYuNjY3IC0yOTQuMzMzLTI0MC03MjdsLTIxMiAtNjQzIC04NSAxNzAKYy00LTMuMzMzLTguMzMzLTcuNjY3LTEzIC0xM2wtMTMtMTNsNzctMTU1IDc3LTE1NmM2NiAxOTkuMzMzIDEzOSA0MTkuNjY3CjIxOSA2NjEgbDIxOCA2NjF6TTcwMiA4MEg0MDAwMDB2NDBINzQyeiI+CjwvcGF0aD4KPC9zdmc+)​

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->x xx  is the sample set of elements, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->x ˉ bar{x}x ˉ  is the
sample mean, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            N
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           N
          </annotation>
</semantics>
</math> -->N NN  is the number of samples and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->δ N delta Nδ N  is
the `correction`  . 

If `keepdim`  is `True`  , the output tensor is of the same size
as `input`  except in the dimension(s) `dim`  where it is of size 1.
Otherwise, `dim`  is squeezed (see [`torch.squeeze()`](torch.squeeze.html#torch.squeeze "torch.squeeze")  ), resulting in the
output tensor having 1 (or `len(dim)`  ) fewer dimension(s). 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the input tensor.
* **dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *of* *ints* *,* *optional*  ) – the dimension or dimensions to reduce.
If `None`  , all dimensions are reduced.

Keyword Arguments
:   * **correction** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) –

    difference between the sample size and sample degrees of freedom.
        Defaults to [Bessel’s correction](https://en.wikipedia.org/wiki/Bessel%27s_correction)  , `correction=1`  .

    Changed in version 2.0:  Previously this argument was called `unbiased`  and was a boolean
        with `True`  corresponding to `correction=1`  and `False`  being `correction=0`  .

* **keepdim** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – whether the output tensor has `dim`  retained or not. Default: `False`  .
* **out** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – the output tensor.

Returns
:   A tuple (std, mean) containing the standard deviation and mean.

Example 

```
>>> a = torch.tensor(
...     [[ 0.2035,  1.2959,  1.8101, -0.4644],
...      [ 1.5027, -0.3270,  0.5905,  0.6538],
...      [-1.5745,  1.3330, -0.5596, -0.6548],
...      [ 0.1264, -0.5080,  1.6420,  0.1992]]
... )  # fmt: skip
>>> torch.std_mean(a, dim=0, keepdim=True)
(tensor([[1.2620, 1.0028, 1.0957, 0.6038]]),
 tensor([[ 0.0645,  0.4485,  0.8707, -0.0665]]))

```

