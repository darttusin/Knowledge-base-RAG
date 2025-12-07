torch.tensordot 
==================================================================

torch. tensordot ( *a*  , *b*  , *dims = 2*  , *out = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/functional.py#L1285) 
:   Returns a contraction of a and b over multiple dimensions. 

[`tensordot`](#torch.tensordot "torch.tensordot")  implements a generalized matrix product. 

Parameters
:   * **a** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Left tensor to contract
* **b** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Right tensor to contract
* **dims** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* *Tuple* *[* *List* *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *,* *List* *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *] or* *List* *[* *List* *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]* *]* *containing two lists* *or* [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – number of dimensions to
contract or explicit lists of dimensions for `a`  and `b`  respectively

When called with a non-negative integer argument `dims`  = <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            d
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           d
          </annotation>
</semantics>
</math> -->d dd  , and
the number of dimensions of `a`  and `b`  is <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            m
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           m
          </annotation>
</semantics>
</math> -->m mm  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            n
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           n
          </annotation>
</semantics>
</math> -->n nn  ,
respectively, [`tensordot()`](#torch.tensordot "torch.tensordot")  computes 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             r
            </mi>
<mrow>
<msub>
<mi>
               i
              </mi>
<mn>
               0
              </mn>
</msub>
<mo separator="true">
              ,
             </mo>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mo separator="true">
              ,
             </mo>
<msub>
<mi>
               i
              </mi>
<mrow>
<mi>
                m
               </mi>
<mo>
                −
               </mo>
<mi>
                d
               </mi>
</mrow>
</msub>
<mo separator="true">
              ,
             </mo>
<msub>
<mi>
               i
              </mi>
<mi>
               d
              </mi>
</msub>
<mo separator="true">
              ,
             </mo>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mo separator="true">
              ,
             </mo>
<msub>
<mi>
               i
              </mi>
<mi>
               n
              </mi>
</msub>
</mrow>
</msub>
<mo>
            =
           </mo>
<munder>
<mo>
             ∑
            </mo>
<mrow>
<msub>
<mi>
               k
              </mi>
<mn>
               0
              </mn>
</msub>
<mo separator="true">
              ,
             </mo>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mo separator="true">
              ,
             </mo>
<msub>
<mi>
               k
              </mi>
<mrow>
<mi>
                d
               </mi>
<mo>
                −
               </mo>
<mn>
                1
               </mn>
</mrow>
</msub>
</mrow>
</munder>
<msub>
<mi>
             a
            </mi>
<mrow>
<msub>
<mi>
               i
              </mi>
<mn>
               0
              </mn>
</msub>
<mo separator="true">
              ,
             </mo>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mo separator="true">
              ,
             </mo>
<msub>
<mi>
               i
              </mi>
<mrow>
<mi>
                m
               </mi>
<mo>
                −
               </mo>
<mi>
                d
               </mi>
</mrow>
</msub>
<mo separator="true">
              ,
             </mo>
<msub>
<mi>
               k
              </mi>
<mn>
               0
              </mn>
</msub>
<mo separator="true">
              ,
             </mo>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mo separator="true">
              ,
             </mo>
<msub>
<mi>
               k
              </mi>
<mrow>
<mi>
                d
               </mi>
<mo>
                −
               </mo>
<mn>
                1
               </mn>
</mrow>
</msub>
</mrow>
</msub>
<mo>
            ×
           </mo>
<msub>
<mi>
             b
            </mi>
<mrow>
<msub>
<mi>
               k
              </mi>
<mn>
               0
              </mn>
</msub>
<mo separator="true">
              ,
             </mo>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mo separator="true">
              ,
             </mo>
<msub>
<mi>
               k
              </mi>
<mrow>
<mi>
                d
               </mi>
<mo>
                −
               </mo>
<mn>
                1
               </mn>
</mrow>
</msub>
<mo separator="true">
              ,
             </mo>
<msub>
<mi>
               i
              </mi>
<mi>
               d
              </mi>
</msub>
<mo separator="true">
              ,
             </mo>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mi mathvariant="normal">
              .
             </mi>
<mo separator="true">
              ,
             </mo>
<msub>
<mi>
               i
              </mi>
<mi>
               n
              </mi>
</msub>
</mrow>
</msub>
<mi mathvariant="normal">
            .
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           r_{i_0,...,i_{m-d}, i_d,...,i_n}
  = sum_{k_0,...,k_{d-1}} a_{i_0,...,i_{m-d},k_0,...,k_{d-1}} times b_{k_0,...,k_{d-1}, i_d,...,i_n}.
          </annotation>
</semantics>
</math> -->
r i 0 , . . . , i m − d , i d , . . . , i n = ∑ k 0 , . . . , k d − 1 a i 0 , . . . , i m − d , k 0 , . . . , k d − 1 × b k 0 , . . . , k d − 1 , i d , . . . , i n . r_{i_0,...,i_{m-d}, i_d,...,i_n}
 = sum_{k_0,...,k_{d-1}} a_{i_0,...,i_{m-d},k_0,...,k_{d-1}} times b_{k_0,...,k_{d-1}, i_d,...,i_n}.

r i 0 ​ , ... , i m − d ​ , i d ​ , ... , i n ​ ​ = k 0 ​ , ... , k d − 1 ​ ∑ ​ a i 0 ​ , ... , i m − d ​ , k 0 ​ , ... , k d − 1 ​ ​ × b k 0 ​ , ... , k d − 1 ​ , i d ​ , ... , i n ​ ​ .

When called with `dims`  of the list form, the given dimensions will be contracted
in place of the last <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            d
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           d
          </annotation>
</semantics>
</math> -->d dd  of `a`  and the first <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            d
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           d
          </annotation>
</semantics>
</math> -->d dd  of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->b bb  . The sizes
in these dimensions must match, but [`tensordot()`](#torch.tensordot "torch.tensordot")  will deal with broadcasted
dimensions. 

Examples: 

```
>>> a = torch.arange(60.).reshape(3, 4, 5)
>>> b = torch.arange(24.).reshape(4, 3, 2)
>>> torch.tensordot(a, b, dims=([1, 0], [0, 1]))
tensor([[4400., 4730.],
        [4532., 4874.],
        [4664., 5018.],
        [4796., 5162.],
        [4928., 5306.]])

>>> a = torch.randn(3, 4, 5, device='cuda')
>>> b = torch.randn(4, 5, 6, device='cuda')
>>> c = torch.tensordot(a, b, dims=2).cpu()
tensor([[ 8.3504, -2.5436,  6.2922,  2.7556, -1.0732,  3.2741],
        [ 3.3161,  0.0704,  5.0187, -0.4079, -4.3126,  4.8744],
        [ 0.8223,  3.9445,  3.2168, -0.2400,  3.4117,  1.7780]])

>>> a = torch.randn(3, 5, 4, 6)
>>> b = torch.randn(6, 4, 5, 3)
>>> torch.tensordot(a, b, dims=([2, 1, 3], [1, 2, 0]))
tensor([[  7.7193,  -2.4867, -10.3204],
        [  1.5513, -14.4737,  -6.5113],
        [ -0.2850,   4.2573,  -3.5997]])

```

