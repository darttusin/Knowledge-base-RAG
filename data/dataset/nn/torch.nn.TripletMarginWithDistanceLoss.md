TripletMarginWithDistanceLoss 
==============================================================================================

*class* torch.nn. TripletMarginWithDistanceLoss ( *** , *distance_function = None*  , *margin = 1.0*  , *swap = False*  , *reduction = 'mean'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py#L1717) 
:   Creates a criterion that measures the triplet loss given input
tensors <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            a
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           a
          </annotation>
</semantics>
</math> -->a aa  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            p
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           p
          </annotation>
</semantics>
</math> -->p pp  , and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->n nn  (representing anchor,
positive, and negative examples, respectively), and a nonnegative,
real-valued function (“distance function”) used to compute the relationship
between the anchor and positive example (“positive distance”) and the
anchor and negative example (“negative distance”). 

The unreduced loss (i.e., with `reduction`  set to `'none'`  )
can be described as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
            ℓ
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
            L
           </mi>
<mo>
            =
           </mo>
<mo stretchy="false">
            {
           </mo>
<msub>
<mi>
             l
            </mi>
<mn>
             1
            </mn>
</msub>
<mo separator="true">
            ,
           </mo>
<mo>
            …
           </mo>
<mo separator="true">
            ,
           </mo>
<msub>
<mi>
             l
            </mi>
<mi>
             N
            </mi>
</msub>
<msup>
<mo stretchy="false">
             }
            </mo>
<mi mathvariant="normal">
             ⊤
            </mi>
</msup>
<mo separator="true">
            ,
           </mo>
<mspace width="1em">
</mspace>
<msub>
<mi>
             l
            </mi>
<mi>
             i
            </mi>
</msub>
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
           ell(a, p, n) = L = {l_1,dots,l_N}^top, quad
l_i = max {d(a_i, p_i) - d(a_i, n_i) + {rm margin}, 0}
          </annotation>
</semantics>
</math> -->
ℓ ( a , p , n ) = L = { l 1 , … , l N } ⊤ , l i = max ⁡ { d ( a i , p i ) − d ( a i , n i ) + m a r g i n , 0 } ell(a, p, n) = L = {l_1,dots,l_N}^top, quad
l_i = max {d(a_i, p_i) - d(a_i, n_i) + {rm margin}, 0}

ℓ ( a , p , n ) = L = { l 1 ​ , … , l N ​ } ⊤ , l i ​ = max { d ( a i ​ , p i ​ ) − d ( a i ​ , n i ​ ) + margin , 0 }

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->N NN  is the batch size; <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->d dd  is a nonnegative, real-valued function
quantifying the closeness of two tensors, referred to as the `distance_function`  ;
and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            m
           </mi>
<mi>
            a
           </mi>
<mi>
            r
           </mi>
<mi>
            g
           </mi>
<mi>
            i
           </mi>
<mi>
            n
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           margin
          </annotation>
</semantics>
</math> -->m a r g i n marginma r g in  is a nonnegative margin representing the minimum difference
between the positive and negative distances that is required for the loss to
be 0. The input tensors have <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->N NN  elements each and can be of any shape
that the distance function can handle. 

If `reduction`  is not `'none'`  (default `'mean'`  ), then: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
            ℓ
           </mi>
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
<mrow>
<mo fence="true">
             {
            </mo>
<mtable columnalign="left left" columnspacing="1em" rowspacing="0.36em">
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mi mathvariant="normal">
                  mean
                 </mi>
<mo>
                  ⁡
                 </mo>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  L
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
<mo separator="true">
                  ,
                 </mo>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mtext>
                  if reduction
                 </mtext>
<mo>
                  =
                 </mo>
<mtext>
                  ‘mean’;
                 </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mi mathvariant="normal">
                  sum
                 </mi>
<mo>
                  ⁡
                 </mo>
<mo stretchy="false">
                  (
                 </mo>
<mi>
                  L
                 </mi>
<mo stretchy="false">
                  )
                 </mo>
<mo separator="true">
                  ,
                 </mo>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mtext>
                  if reduction
                 </mtext>
<mo>
                  =
                 </mo>
<mtext>
                  ‘sum’.
                 </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           ell(x, y) =
begin{cases}
    operatorname{mean}(L), &amp;  text{if reduction} = text{`mean';}
    operatorname{sum}(L),  &amp;  text{if reduction} = text{`sum'.}
end{cases}
          </annotation>
</semantics>
</math> -->
ℓ ( x , y ) = { mean ⁡ ( L ) , if reduction = ‘mean’; sum ⁡ ( L ) , if reduction = ‘sum’. ell(x, y) =
begin{cases}
 operatorname{mean}(L), & text{if reduction} = text{`mean';}
 operatorname{sum}(L), & text{if reduction} = text{`sum'.}
end{cases}

ℓ ( x , y ) = { mean ( L ) , sum ( L ) , ​ if reduction = ‘mean’; if reduction = ‘sum’. ​

See also [`TripletMarginLoss`](torch.nn.TripletMarginLoss.html#torch.nn.TripletMarginLoss "torch.nn.TripletMarginLoss")  , which computes the triplet
loss for input tensors using the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             l
            </mi>
<mi>
             p
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           l_p
          </annotation>
</semantics>
</math> -->l p l_pl p ​  distance as the distance function. 

Parameters
:   * **distance_function** ( *Callable* *,* *optional*  ) – A nonnegative, real-valued function that
quantifies the closeness of two tensors. If not specified, *nn.PairwiseDistance* will be used. Default: `None`
* **margin** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – A nonnegative margin representing the minimum difference
between the positive and negative distances required for the loss to be 0. Larger
margins penalize cases where the negative examples are not distant enough from the
anchors, relative to the positives. Default: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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

* **swap** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Whether to use the distance swap described in the paper *Learning shallow convolutional feature descriptors with triplet losses* by
V. Balntas, E. Riba et al. If True, and if the positive example is closer to the
negative example than the anchor is, swaps the positive example and the anchor in
the loss computation. Default: `False`  .
* **reduction** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – Specifies the (optional) reduction to apply to the output: `'none'`  | `'mean'`  | `'sum'`  . `'none'`  : no reduction will be applied, `'mean'`  : the sum of the output will be divided by the number of
elements in the output, `'sum'`  : the output will be summed. Default: `'mean'`

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
<mo>
                ∗
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, *)
              </annotation>
</semantics>
</math> -->( N , ∗ ) (N, *)( N , ∗ )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
                ∗
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               *
              </annotation>
</semantics>
</math> -->∗ *∗  represents any number of additional dimensions
as supported by the distance function.

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
</math> -->( N ) (N)( N )  if `reduction`  is `'none'`  , or a scalar
otherwise.

Examples: 

```
>>> # Initialize embeddings
>>> embedding = nn.Embedding(1000, 128)
>>> anchor_ids = torch.randint(0, 1000, (1,))
>>> positive_ids = torch.randint(0, 1000, (1,))
>>> negative_ids = torch.randint(0, 1000, (1,))
>>> anchor = embedding(anchor_ids)
>>> positive = embedding(positive_ids)
>>> negative = embedding(negative_ids)
>>>
>>> # Built-in Distance Function
>>> triplet_loss = 
>>>     nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())
>>> output = triplet_loss(anchor, positive, negative)
>>> output.backward()
>>>
>>> # Custom Distance Function
>>> def l_infinity(x1, x2):
>>>     return torch.max(torch.abs(x1 - x2), dim=1).values
>>>
>>> triplet_loss = (
>>>     nn.TripletMarginWithDistanceLoss(distance_function=l_infinity, margin=1.5))
>>> output = triplet_loss(anchor, positive, negative)
>>> output.backward()
>>>
>>> # Custom Distance Function (Lambda)
>>> triplet_loss = (
>>>     nn.TripletMarginWithDistanceLoss(
>>>         distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)))
>>> output = triplet_loss(anchor, positive, negative)
>>> output.backward()

```

Reference:
:   V. Balntas, et al.: Learning shallow convolutional feature descriptors with triplet losses: [https://bmva-archive.org.uk/bmvc/2016/papers/paper119/index.html](https://bmva-archive.org.uk/bmvc/2016/papers/paper119/index.html)

