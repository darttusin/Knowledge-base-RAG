TripletMarginLoss 
======================================================================

*class* torch.nn. TripletMarginLoss ( *margin = 1.0*  , *p = 2.0*  , *eps = 1e-06*  , *swap = False*  , *size_average = None*  , *reduce = None*  , *reduction = 'mean'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py#L1608) 
:   Creates a criterion that measures the triplet loss given an input
tensors <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            x
           </mi>
<mn>
            1
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           x1
          </annotation>
</semantics>
</math> -->x 1 x1x 1  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            x
           </mi>
<mn>
            2
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           x2
          </annotation>
</semantics>
</math> -->x 2 x2x 2  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            x
           </mi>
<mn>
            3
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           x3
          </annotation>
</semantics>
</math> -->x 3 x3x 3  and a margin with a value greater than <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
            0
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           0
          </annotation>
</semantics>
</math> -->0 00  .
This is used for measuring a relative similarity between samples. A triplet
is composed by *a* , *p* and *n* (i.e., *anchor* , *positive examples* and *negative
examples* respectively). The shapes of all input tensors should be <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N , D ) (N, D)( N , D )  . 

The distance swap is described in detail in the paper [Learning shallow
convolutional feature descriptors with triplet losses](https://bmva-archive.org.uk/bmvc/2016/papers/paper119/index.html)  by
V. Balntas, E. Riba et al. 

The loss function for each sample in the mini-batch is: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            L
           </mi>
<mo stretchy="false">
            (
           </mo>
<mi>
            a
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            p
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
<mo>
            =
           </mo>
<mi>
            max
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            {
           </mo>
<mi>
            d
           </mi>
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             a
            </mi>
<mi>
             i
            </mi>
</msub>
<mo separator="true">
            ,
           </mo>
<msub>
<mi>
             p
            </mi>
<mi>
             i
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
<mo>
            −
           </mo>
<mi>
            d
           </mi>
<mo stretchy="false">
            (
           </mo>
<msub>
<mi>
             a
            </mi>
<mi>
             i
            </mi>
</msub>
<mo separator="true">
            ,
           </mo>
<msub>
<mi>
             n
            </mi>
<mi>
             i
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
<mo>
            +
           </mo>
<mrow>
<mi mathvariant="normal">
             m
            </mi>
<mi mathvariant="normal">
             a
            </mi>
<mi mathvariant="normal">
             r
            </mi>
<mi mathvariant="normal">
             g
            </mi>
<mi mathvariant="normal">
             i
            </mi>
<mi mathvariant="normal">
             n
            </mi>
</mrow>
<mo separator="true">
            ,
           </mo>
<mn>
            0
           </mn>
<mo stretchy="false">
            }
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           L(a, p, n) = max {d(a_i, p_i) - d(a_i, n_i) + {rm margin}, 0}
          </annotation>
</semantics>
</math> -->
L ( a , p , n ) = max ⁡ { d ( a i , p i ) − d ( a i , n i ) + m a r g i n , 0 } L(a, p, n) = max {d(a_i, p_i) - d(a_i, n_i) + {rm margin}, 0}

L ( a , p , n ) = max { d ( a i ​ , p i ​ ) − d ( a i ​ , n i ​ ) + margin , 0 }

where 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            d
           </mi>
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
<mo separator="true">
            ,
           </mo>
<msub>
<mi>
             y
            </mi>
<mi>
             i
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<msub>
<mrow>
<mo fence="true">
              ∥
             </mo>
<msub>
<mi mathvariant="bold">
               x
              </mi>
<mi>
               i
              </mi>
</msub>
<mo>
              −
             </mo>
<msub>
<mi mathvariant="bold">
               y
              </mi>
<mi>
               i
              </mi>
</msub>
<mo fence="true">
              ∥
             </mo>
</mrow>
<mi>
             p
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           d(x_i, y_i) = leftlVert {bf x}_i - {bf y}_i rightrVert_p
          </annotation>
</semantics>
</math> -->
d ( x i , y i ) = ∥ x i − y i ∥ p d(x_i, y_i) = leftlVert {bf x}_i - {bf y}_i rightrVert_p

d ( x i ​ , y i ​ ) = ∥ x i ​ − y i ​ ∥ p ​

The norm is calculated using the specified p value and a small constant <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            ε
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           varepsilon
          </annotation>
</semantics>
</math> -->ε varepsilonε  is
added for numerical stability. 

See also [`TripletMarginWithDistanceLoss`](torch.nn.TripletMarginWithDistanceLoss.html#torch.nn.TripletMarginWithDistanceLoss "torch.nn.TripletMarginWithDistanceLoss")  , which computes the
triplet margin loss for input tensors using a custom distance function. 

Parameters
:   * **margin** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – Default: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
                1
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               1
              </annotation>
</semantics>
</math> -->1 11  .

* **p** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – The norm degree for pairwise distance. Default: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
                2
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               2
              </annotation>
</semantics>
</math> -->2 22  .

* **eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – Small constant for numerical stability. Default: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
                1
               </mn>
<mi>
                e
               </mi>
<mo>
                −
               </mo>
<mn>
                6
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               1e-6
              </annotation>
</semantics>
</math> -->1 e − 6 1e-61 e − 6  .

* **swap** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – The distance swap is described in detail in the paper *Learning shallow convolutional feature descriptors with triplet losses* by
V. Balntas, E. Riba et al. Default: `False`  .
* **size_average** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ). By default,
the losses are averaged over each loss element in the batch. Note that for
some losses, there are multiple elements per sample. If the field `size_average`  is set to `False`  , the losses are instead summed for each minibatch. Ignored
when `reduce`  is `False`  . Default: `True`
* **reduce** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ). By default, the
losses are averaged or summed over observations for each minibatch depending
on `size_average`  . When `reduce`  is `False`  , returns a loss per
batch element instead and ignores `size_average`  . Default: `True`
* **reduction** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – Specifies the reduction to apply to the output: `'none'`  | `'mean'`  | `'sum'`  . `'none'`  : no reduction will be applied, `'mean'`  : the sum of the output will be divided by the number of
elements in the output, `'sum'`  : the output will be summed. Note: `size_average`  and `reduce`  are in the process of being deprecated, and in the meantime,
specifying either of those two args will override `reduction`  . Default: `'mean'`

Shape:
:   * Input: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( D ) (D)( D )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                D
               </mi>
</mrow>
<annotation encoding="application/x-tex">
               D
              </annotation>
</semantics>
</math> -->D DD  is the vector dimension.

* Output: A Tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N ) (N)( N )  if `reduction`  is `'none'`  and
input shape is <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N , D ) (N, D)( N , D )  ; a scalar otherwise.

Examples: 

```
>>> triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
>>> anchor = torch.randn(100, 128, requires_grad=True)
>>> positive = torch.randn(100, 128, requires_grad=True)
>>> negative = torch.randn(100, 128, requires_grad=True)
>>> output = triplet_loss(anchor, positive, negative)
>>> output.backward()

```

