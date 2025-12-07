torch.svd_lowrank 
=======================================================================

torch. svd_lowrank ( *A*  , *q = 6*  , *niter = 2*  , *M = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_lowrank.py#L86) 
:   Return the singular value decomposition `(U, S, V)`  of a matrix,
batches of matrices, or a sparse matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  such that <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            A
           </mi>
<mo>
            ≈
           </mo>
<mi>
            U
           </mi>
<mi mathvariant="normal">
            diag
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<mi>
            S
           </mi>
<mo stretchy="false">
            )
           </mo>
<msup>
<mi>
             V
            </mi>
<mtext>
             H
            </mtext>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           A approx U operatorname{diag}(S) V^{text{H}}
          </annotation>
</semantics>
</math> -->A ≈ U diag ⁡ ( S ) V H A approx U operatorname{diag}(S) V^{text{H}}A ≈ U diag ( S ) V H  . In case <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->M MM  is given, then
SVD is computed for the matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            A
           </mi>
<mo>
            −
           </mo>
<mi>
            M
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           A - M
          </annotation>
</semantics>
</math> -->A − M A - MA − M  . 

Note 

The implementation is based on the Algorithm 5.1 from
Halko et al., 2009.

Note 

For an adequate approximation of a k-rank matrix <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->A AA  , where k is not known in advance but could be
estimated, the number of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->Q QQ  columns, q, can be
choosen according to the following criteria: in general, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             k
            </mi>
<mo>
             &lt;
            </mo>
<mo>
             =
            </mo>
<mi>
             q
            </mi>
<mo>
             &lt;
            </mo>
<mo>
             =
            </mo>
<mi>
             m
            </mi>
<mi>
             i
            </mi>
<mi>
             n
            </mi>
<mo stretchy="false">
             (
            </mo>
<mn>
             2
            </mn>
<mo>
             ∗
            </mo>
<mi>
             k
            </mi>
<mo separator="true">
             ,
            </mo>
<mi>
             m
            </mi>
<mo separator="true">
             ,
            </mo>
<mi>
             n
            </mi>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            k &lt;= q &lt;= min(2*k, m, n)
           </annotation>
</semantics>
</math> -->k < = q < = m i n ( 2 ∗ k , m , n ) k <= q <= min(2*k, m, n)k <= q <= min ( 2 ∗ k , m , n )  . For large low-rank
matrices, take <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             q
            </mi>
<mo>
             =
            </mo>
<mi>
             k
            </mi>
<mo>
             +
            </mo>
<mn>
             5..10
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            q = k + 5..10
           </annotation>
</semantics>
</math> -->q = k + 5..10 q = k + 5..10q = k + 5..10  . If k is
relatively small compared to <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             m
            </mi>
<mi>
             i
            </mi>
<mi>
             n
            </mi>
<mo stretchy="false">
             (
            </mo>
<mi>
             m
            </mi>
<mo separator="true">
             ,
            </mo>
<mi>
             n
            </mi>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            min(m, n)
           </annotation>
</semantics>
</math> -->m i n ( m , n ) min(m, n)min ( m , n )  , choosing <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             q
            </mi>
<mo>
             =
            </mo>
<mi>
             k
            </mi>
<mo>
             +
            </mo>
<mn>
             0..2
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            q = k + 0..2
           </annotation>
</semantics>
</math> -->q = k + 0..2 q = k + 0..2q = k + 0..2  may be sufficient.

Note 

This is a randomized method. To obtain repeatable results,
set the seed for the pseudorandom number generator

Note 

In general, use the full-rank SVD implementation [`torch.linalg.svd()`](torch.linalg.svd.html#torch.linalg.svd "torch.linalg.svd")  for dense matrices due to its 10x
higher performance characteristics. The low-rank SVD
will be useful for huge sparse matrices that [`torch.linalg.svd()`](torch.linalg.svd.html#torch.linalg.svd "torch.linalg.svd")  cannot handle.

Args::
:   A (Tensor): the input tensor of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mo>
              ∗
             </mo>
<mo separator="true">
              ,
             </mo>
<mi>
              m
             </mi>
<mo separator="true">
              ,
             </mo>
<mi>
              n
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             (*, m, n)
            </annotation>
</semantics>
</math> -->( ∗ , m , n ) (*, m, n)( ∗ , m , n ) 

q (int, optional): a slightly overestimated rank of A. 

niter (int, optional): the number of subspace iterations to
:   conduct; niter must be a nonnegative
integer, and defaults to 2

M (Tensor, optional): the input tensor’s mean of size
:   <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo separator="true">
                ,
               </mo>
<mi>
                m
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                n
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (*, m, n)
              </annotation>
</semantics>
</math> -->( ∗ , m , n ) (*, m, n)( ∗ , m , n )  , which will be broadcasted
to the size of A in this function.

References::
:   * Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
structure with randomness: probabilistic algorithms for
constructing approximate matrix decompositions,
arXiv:0909.4061 [math.NA; math.PR], 2009 (available at [arXiv](https://arxiv.org/abs/0909.4061)  ).

Return type
:   [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [ [torch.Tensor](../tensors.html#torch.Tensor "torch.Tensor")  , [torch.Tensor](../tensors.html#torch.Tensor "torch.Tensor")  , [torch.Tensor](../tensors.html#torch.Tensor "torch.Tensor")  ]

