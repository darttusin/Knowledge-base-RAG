NLLLoss 
==================================================

*class* torch.nn. NLLLoss ( *weight = None*  , *size_average = None*  , *ignore_index = -100*  , *reduce = None*  , *reduction = 'mean'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py#L132) 
:   The negative log likelihood loss. It is useful to train a classification
problem with *C* classes. 

If provided, the optional argument `weight`  should be a 1D Tensor assigning
weight to each of the classes. This is particularly useful when you have an
unbalanced training set. 

The *input* given through a forward call is expected to contain
log-probabilities of each class. *input* has to be a Tensor of size either <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
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
<mi>
            i
           </mi>
<mi>
            b
           </mi>
<mi>
            a
           </mi>
<mi>
            t
           </mi>
<mi>
            c
           </mi>
<mi>
            h
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            C
           </mi>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (minibatch, C)
          </annotation>
</semantics>
</math> -->( m i n i b a t c h , C ) (minibatch, C)( miniba t c h , C )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
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
<mi>
            i
           </mi>
<mi>
            b
           </mi>
<mi>
            a
           </mi>
<mi>
            t
           </mi>
<mi>
            c
           </mi>
<mi>
            h
           </mi>
<mo separator="true">
            ,
           </mo>
<mi>
            C
           </mi>
<mo separator="true">
            ,
           </mo>
<msub>
<mi>
             d
            </mi>
<mn>
             1
            </mn>
</msub>
<mo separator="true">
            ,
           </mo>
<msub>
<mi>
             d
            </mi>
<mn>
             2
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
             d
            </mi>
<mi>
             K
            </mi>
</msub>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           (minibatch, C, d_1, d_2, ..., d_K)
          </annotation>
</semantics>
</math> -->( m i n i b a t c h , C , d 1 , d 2 , . . . , d K ) (minibatch, C, d_1, d_2, ..., d_K)( miniba t c h , C , d 1 ​ , d 2 ​ , ... , d K ​ )  with <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            K
           </mi>
<mo>
            ≥
           </mo>
<mn>
            1
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           K geq 1
          </annotation>
</semantics>
</math> -->K ≥ 1 K geq 1K ≥ 1  for the *K* -dimensional case. The latter is useful for
higher dimension inputs, such as computing NLL loss per-pixel for 2D images. 

Obtaining log-probabilities in a neural network is easily achieved by
adding a *LogSoftmax* layer in the last layer of your network.
You may use *CrossEntropyLoss* instead, if you prefer not to add an extra
layer. 

The *target* that this loss expects should be a class index in the range <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            [
           </mo>
<mn>
            0
           </mn>
<mo separator="true">
            ,
           </mo>
<mi>
            C
           </mi>
<mo>
            −
           </mo>
<mn>
            1
           </mn>
<mo stretchy="false">
            ]
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           [0, C-1]
          </annotation>
</semantics>
</math> -->[ 0 , C − 1 ] [0, C-1][ 0 , C − 1 ]  where *C = number of classes* ; if *ignore_index* is specified, this loss also accepts
this class index (this index may not necessarily be in the class range). 

The unreduced (i.e. with `reduction`  set to `'none'`  ) loss can be described as: 

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
<mspace linebreak="newline">
</mspace>
<msub>
<mi>
             l
            </mi>
<mi>
             n
            </mi>
</msub>
<mo>
            =
           </mo>
<mo>
            −
           </mo>
<msub>
<mi>
             w
            </mi>
<msub>
<mi>
              y
             </mi>
<mi>
              n
             </mi>
</msub>
</msub>
<msub>
<mi>
             x
            </mi>
<mrow>
<mi>
              n
             </mi>
<mo separator="true">
              ,
             </mo>
<msub>
<mi>
               y
              </mi>
<mi>
               n
              </mi>
</msub>
</mrow>
</msub>
<mo separator="true">
            ,
           </mo>
<mspace linebreak="newline">
</mspace>
<msub>
<mi>
             w
            </mi>
<mi>
             c
            </mi>
</msub>
<mo>
            =
           </mo>
<mtext>
            weight
           </mtext>
<mo stretchy="false">
            [
           </mo>
<mi>
            c
           </mi>
<mo stretchy="false">
            ]
           </mo>
<mo>
            ⋅
           </mo>
<mn mathvariant="double-struck">
            1
           </mn>
<mo stretchy="false">
            {
           </mo>
<mi>
            c
           </mi>
<mo>
            ≠
           </mo>
<mtext>
            ignore_index
           </mtext>
<mo stretchy="false">
            }
           </mo>
<mo separator="true">
            ,
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           ell(x, y) = L = {l_1,dots,l_N}^top, 
l_n = - w_{y_n} x_{n,y_n}, 
w_{c} = text{weight}[c] cdot mathbb{1}{c not= text{ignore_index}},
          </annotation>
</semantics>
</math> -->
ℓ ( x , y ) = L = { l 1 , … , l N } ⊤ , l n = − w y n x n , y n , w c = weight [ c ] ⋅ 1 { c ≠ ignore_index } , ell(x, y) = L = {l_1,dots,l_N}^top, 
l_n = - w_{y_n} x_{n,y_n}, 
w_{c} = text{weight}[c] cdot mathbb{1}{c not= text{ignore_index}},

ℓ ( x , y ) = L = { l 1 ​ , … , l N ​ } ⊤ , l n ​ = − w y n ​ ​ x n , y n ​ ​ , w c ​ = weight [ c ] ⋅ 1 { c  = ignore_index } ,

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
</math> -->x xx  is the input, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->y yy  is the target, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->w ww  is the weight, and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->N NN  is the batch size. If `reduction`  is not `'none'`  (default `'mean'`  ), then 

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
<msubsup>
<mo>
                   ∑
                  </mo>
<mrow>
<mi>
                    n
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
<mfrac>
<mn>
                   1
                  </mn>
<mrow>
<msubsup>
<mo>
                     ∑
                    </mo>
<mrow>
<mi>
                      n
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
<msub>
<mi>
                      y
                     </mi>
<mi>
                      n
                     </mi>
</msub>
</msub>
</mrow>
</mfrac>
<msub>
<mi>
                   l
                  </mi>
<mi>
                   n
                  </mi>
</msub>
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
<msubsup>
<mo>
                   ∑
                  </mo>
<mrow>
<mi>
                    n
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
                   l
                  </mi>
<mi>
                   n
                  </mi>
</msub>
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
           ell(x, y) = begin{cases}
    sum_{n=1}^N frac{1}{sum_{n=1}^N w_{y_n}} l_n, &amp;
    text{if reduction} = text{`mean';}
    sum_{n=1}^N l_n,  &amp;
    text{if reduction} = text{`sum'.}
end{cases}
          </annotation>
</semantics>
</math> -->
ℓ ( x , y ) = { ∑ n = 1 N 1 ∑ n = 1 N w y n l n , if reduction = ‘mean’; ∑ n = 1 N l n , if reduction = ‘sum’. ell(x, y) = begin{cases}
 sum_{n=1}^N frac{1}{sum_{n=1}^N w_{y_n}} l_n, &
 text{if reduction} = text{`mean';}
 sum_{n=1}^N l_n, &
 text{if reduction} = text{`sum'.}
end{cases}

ℓ ( x , y ) = { ∑ n = 1 N ​ ∑ n = 1 N ​ w y n ​ ​ 1 ​ l n ​ , ∑ n = 1 N ​ l n ​ , ​ if reduction = ‘mean’; if reduction = ‘sum’. ​

Parameters
:   * **weight** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – a manual rescaling weight given to each
class. If given, it has to be a Tensor of size *C* . Otherwise, it is
treated as if having all ones.
* **size_average** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ). By default,
the losses are averaged over each loss element in the batch. Note that for
some losses, there are multiple elements per sample. If the field `size_average`  is set to `False`  , the losses are instead summed for each minibatch. Ignored
when `reduce`  is `False`  . Default: `None`
* **ignore_index** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – Specifies a target value that is ignored
and does not contribute to the input gradient. When `size_average`  is `True`  , the loss is averaged over
non-ignored targets.
* **reduce** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ). By default, the
losses are averaged or summed over observations for each minibatch depending
on `size_average`  . When `reduce`  is `False`  , returns a loss per
batch element instead and ignores `size_average`  . Default: `None`
* **reduction** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – Specifies the reduction to apply to the output: `'none'`  | `'mean'`  | `'sum'`  . `'none'`  : no reduction will
be applied, `'mean'`  : the weighted mean of the output is taken, `'sum'`  : the output will be summed. Note: `size_average`  and `reduce`  are in the process of being deprecated, and in
the meantime, specifying either of those two args will override `reduction`  . Default: `'mean'`

Shape::
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
                C
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C)
              </annotation>
</semantics>
</math> -->( N , C ) (N, C)( N , C )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                C
               </mi>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (C)
              </annotation>
</semantics>
</math> -->( C ) (C)( C )  , where *C = number of classes* , *N = batch size* , or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                C
               </mi>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 d
                </mi>
<mn>
                 1
                </mn>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 d
                </mi>
<mn>
                 2
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
                 d
                </mi>
<mi>
                 K
                </mi>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, C, d_1, d_2, ..., d_K)
              </annotation>
</semantics>
</math> -->( N , C , d 1 , d 2 , . . . , d K ) (N, C, d_1, d_2, ..., d_K)( N , C , d 1 ​ , d 2 ​ , ... , d K ​ )  with <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                K
               </mi>
<mo>
                ≥
               </mo>
<mn>
                1
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               K geq 1
              </annotation>
</semantics>
</math> -->K ≥ 1 K geq 1K ≥ 1  in the case of *K* -dimensional loss.

* Target: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N ) (N)( N )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               ()
              </annotation>
</semantics>
</math> -->( ) ()( )  , where each value is <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
                0
               </mn>
<mo>
                ≤
               </mo>
<mtext>
                targets
               </mtext>
<mo stretchy="false">
                [
               </mo>
<mi>
                i
               </mi>
<mo stretchy="false">
                ]
               </mo>
<mo>
                ≤
               </mo>
<mi>
                C
               </mi>
<mo>
                −
               </mo>
<mn>
                1
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               0 leq text{targets}[i] leq C-1
              </annotation>
</semantics>
</math> -->0 ≤ targets [ i ] ≤ C − 1 0 leq text{targets}[i] leq C-10 ≤ targets [ i ] ≤ C − 1  , or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<msub>
<mi>
                 d
                </mi>
<mn>
                 1
                </mn>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 d
                </mi>
<mn>
                 2
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
                 d
                </mi>
<mi>
                 K
                </mi>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, d_1, d_2, ..., d_K)
              </annotation>
</semantics>
</math> -->( N , d 1 , d 2 , . . . , d K ) (N, d_1, d_2, ..., d_K)( N , d 1 ​ , d 2 ​ , ... , d K ​ )  with <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                K
               </mi>
<mo>
                ≥
               </mo>
<mn>
                1
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               K geq 1
              </annotation>
</semantics>
</math> -->K ≥ 1 K geq 1K ≥ 1  in the case of
K-dimensional loss.

* Output: If `reduction`  is `'none'`  , shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N ) (N)( N )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<msub>
<mi>
                 d
                </mi>
<mn>
                 1
                </mn>
</msub>
<mo separator="true">
                ,
               </mo>
<msub>
<mi>
                 d
                </mi>
<mn>
                 2
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
                 d
                </mi>
<mi>
                 K
                </mi>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, d_1, d_2, ..., d_K)
              </annotation>
</semantics>
</math> -->( N , d 1 , d 2 , . . . , d K ) (N, d_1, d_2, ..., d_K)( N , d 1 ​ , d 2 ​ , ... , d K ​ )  with <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
                K
               </mi>
<mo>
                ≥
               </mo>
<mn>
                1
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               K geq 1
              </annotation>
</semantics>
</math> -->K ≥ 1 K geq 1K ≥ 1  in the case of K-dimensional loss.
Otherwise, scalar.

Examples 

```
>>> log_softmax = nn.LogSoftmax(dim=1)
>>> loss_fn = nn.NLLLoss()
>>> # input to NLLLoss is of size N x C = 3 x 5
>>> input = torch.randn(3, 5, requires_grad=True)
>>> # each element in target must have 0 <= value < C
>>> target = torch.tensor([1, 0, 4])
>>> loss = loss_fn(log_softmax(input), target)
>>> loss.backward()
>>>
>>>
>>> # 2D loss example (used, for example, with image inputs)
>>> N, C = 5, 4
>>> loss_fn = nn.NLLLoss()
>>> data = torch.randn(N, 16, 10, 10)
>>> conv = nn.Conv2d(16, C, (3, 3))
>>> log_softmax = nn.LogSoftmax(dim=1)
>>> # output of conv forward is of shape [N, C, 8, 8]
>>> output = log_softmax(conv(data))
>>> # each element in target must have 0 <= value < C
>>> target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
>>> # input to NLLLoss is of size N x C x height (8) x width (8)
>>> loss = loss_fn(output, target)
>>> loss.backward()

```

