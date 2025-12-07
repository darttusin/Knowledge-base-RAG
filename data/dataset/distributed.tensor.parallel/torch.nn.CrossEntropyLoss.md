CrossEntropyLoss 
====================================================================

*class* torch.nn. CrossEntropyLoss ( *weight = None*  , *size_average = None*  , *ignore_index = -100*  , *reduce = None*  , *reduction = 'mean'*  , *label_smoothing = 0.0* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py#L1158) 
:   This criterion computes the cross entropy loss between input logits
and target. 

It is useful when training a classification problem with *C* classes.
If provided, the optional argument `weight`  should be a 1D *Tensor* assigning weight to each of the classes.
This is particularly useful when you have an unbalanced training set. 

The *input* is expected to contain the unnormalized logits for each class (which do *not* need
to be positive or sum to 1, in general). *input* has to be a Tensor of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( C ) (C)( C )  for unbatched input, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->K ≥ 1 K geq 1K ≥ 1  for the *K* -dimensional case. The last being useful for higher dimension inputs, such
as computing cross entropy loss per-pixel for 2D images. 

The *target* that this criterion expects should contain either: 

* Class indices in the range <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mo stretchy="false">
              )
             </mo>
</mrow>
<annotation encoding="application/x-tex">
             [0, C)
            </annotation>
</semantics>
</math> -->[ 0 , C ) [0, C)[ 0 , C )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
              C
             </mi>
</mrow>
<annotation encoding="application/x-tex">
             C
            </annotation>
</semantics>
</math> -->C CC  is the number of classes; if *ignore_index* is specified, this loss also accepts this class index (this index
may not necessarily be in the class range). The unreduced (i.e. with `reduction`  set to `'none'`  ) loss for this case can be described as:

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
    <mspace width="1em">
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
    <mi>
                  log
                 </mi>
    <mo>
                  ⁡
                 </mo>
    <mfrac>
    <mrow>
    <mi>
                    exp
                   </mi>
    <mo>
                    ⁡
                   </mo>
    <mo stretchy="false">
                    (
                   </mo>
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
    <mo stretchy="false">
                    )
                   </mo>
    </mrow>
    <mrow>
    <munderover>
    <mo>
                     ∑
                    </mo>
    <mrow>
    <mi>
                      c
                     </mi>
    <mo>
                      =
                     </mo>
    <mn>
                      1
                     </mn>
    </mrow>
    <mi>
                     C
                    </mi>
    </munderover>
    <mi>
                    exp
                   </mi>
    <mo>
                    ⁡
                   </mo>
    <mo stretchy="false">
                    (
                   </mo>
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
    <mi>
                      c
                     </mi>
    </mrow>
    </msub>
    <mo stretchy="false">
                    )
                   </mo>
    </mrow>
    </mfrac>
    <mo>
                  ⋅
                 </mo>
    <mn mathvariant="double-struck">
                  1
                 </mn>
    <mo stretchy="false">
                  {
                 </mo>
    <msub>
    <mi>
                   y
                  </mi>
    <mi>
                   n
                  </mi>
    </msub>
    <mo>
                  ≠
                 </mo>
    <mtext>
                  ignore_index
                 </mtext>
    <mo stretchy="false">
                  }
                 </mo>
    </mrow>
    <annotation encoding="application/x-tex">
                 ell(x, y) = L = {l_1,dots,l_N}^top, quad
    l_n = - w_{y_n} log frac{exp(x_{n,y_n})}{sum_{c=1}^C exp(x_{n,c})}
    cdot mathbb{1}{y_n not= text{ignore_index}}
                </annotation>
    </semantics>
    </math> -->
    ℓ ( x , y ) = L = { l 1 , … , l N } ⊤ , l n = − w y n log ⁡ exp ⁡ ( x n , y n ) ∑ c = 1 C exp ⁡ ( x n , c ) ⋅ 1 { y n ≠ ignore_index } ell(x, y) = L = {l_1,dots,l_N}^top, quad
    l_n = - w_{y_n} log frac{exp(x_{n,y_n})}{sum_{c=1}^C exp(x_{n,c})}
    cdot mathbb{1}{y_n not= text{ignore_index}}

    ℓ ( x , y ) = L = { l 1 ​ , … , l N ​ } ⊤ , l n ​ = − w y n ​ ​ lo g ∑ c = 1 C ​ exp ( x n , c ​ ) exp ( x n , y n ​ ​ ) ​ ⋅ 1 { y n ​  = ignore_index }

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
        </math> -->w ww  is the weight, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
        <semantics>
        <mrow>
        <mi>
                      C
                     </mi>
        </mrow>
        <annotation encoding="application/x-tex">
                     C
                    </annotation>
        </semantics>
        </math> -->C CC  is the number of classes, and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
        </math> -->N NN  spans the minibatch dimension as well as <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
        <semantics>
        <mrow>
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
                       k
                      </mi>
        </msub>
        </mrow>
        <annotation encoding="application/x-tex">
                     d_1, ..., d_k
                    </annotation>
        </semantics>
        </math> -->d 1 , . . . , d k d_1, ..., d_kd 1 ​ , ... , d k ​  for the *K* -dimensional case. If `reduction`  is not `'none'`  (default `'mean'`  ), then

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
        <mo>
                              ⋅
                             </mo>
        <mn mathvariant="double-struck">
                              1
                             </mn>
        <mo stretchy="false">
                              {
                             </mo>
        <msub>
        <mi>
                               y
                              </mi>
        <mi>
                               n
                              </mi>
        </msub>
        <mo>
                              ≠
                             </mo>
        <mtext>
                              ignore_index
                             </mtext>
        <mo stretchy="false">
                              }
                             </mo>
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
            sum_{n=1}^N frac{1}{sum_{n=1}^N w_{y_n} cdot mathbb{1}{y_n not= text{ignore_index}}} l_n, &amp;
             text{if reduction} = text{`mean';}
              sum_{n=1}^N l_n,  &amp;
              text{if reduction} = text{`sum'.}
          end{cases}
                    </annotation>
        </semantics>
        </math> -->
        ℓ ( x , y ) = { ∑ n = 1 N 1 ∑ n = 1 N w y n ⋅ 1 { y n ≠ ignore_index } l n , if reduction = ‘mean’; ∑ n = 1 N l n , if reduction = ‘sum’. ell(x, y) = begin{cases}
         sum_{n=1}^N frac{1}{sum_{n=1}^N w_{y_n} cdot mathbb{1}{y_n not= text{ignore_index}}} l_n, &
         text{if reduction} = text{`mean';}
         sum_{n=1}^N l_n, &
         text{if reduction} = text{`sum'.}
         end{cases}

    ℓ ( x , y ) = { ∑ n = 1 N ​ ∑ n = 1 N ​ w y n ​ ​ ⋅ 1 { y n ​  = ignore_index } 1 ​ l n ​ , ∑ n = 1 N ​ l n ​ , ​ if reduction = ‘mean’; if reduction = ‘sum’. ​

    Note that this case is equivalent to applying [`LogSoftmax`](torch.nn.LogSoftmax.html#torch.nn.LogSoftmax "torch.nn.LogSoftmax")  on an input, followed by [`NLLLoss`](torch.nn.NLLLoss.html#torch.nn.NLLLoss "torch.nn.NLLLoss")  .

* Probabilities for each class; useful when labels beyond a single class per minibatch item
are required, such as for blended labels, label smoothing, etc. The unreduced (i.e. with `reduction`  set to `'none'`  ) loss for this case can be described as:

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
    <mspace width="1em">
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
    <munderover>
    <mo>
                   ∑
                  </mo>
    <mrow>
    <mi>
                    c
                   </mi>
    <mo>
                    =
                   </mo>
    <mn>
                    1
                   </mn>
    </mrow>
    <mi>
                   C
                  </mi>
    </munderover>
    <msub>
    <mi>
                   w
                  </mi>
    <mi>
                   c
                  </mi>
    </msub>
    <mi>
                  log
                 </mi>
    <mo>
                  ⁡
                 </mo>
    <mfrac>
    <mrow>
    <mi>
                    exp
                   </mi>
    <mo>
                    ⁡
                   </mo>
    <mo stretchy="false">
                    (
                   </mo>
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
    <mi>
                      c
                     </mi>
    </mrow>
    </msub>
    <mo stretchy="false">
                    )
                   </mo>
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
                     C
                    </mi>
    </munderover>
    <mi>
                    exp
                   </mi>
    <mo>
                    ⁡
                   </mo>
    <mo stretchy="false">
                    (
                   </mo>
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
    <mi>
                      i
                     </mi>
    </mrow>
    </msub>
    <mo stretchy="false">
                    )
                   </mo>
    </mrow>
    </mfrac>
    <msub>
    <mi>
                   y
                  </mi>
    <mrow>
    <mi>
                    n
                   </mi>
    <mo separator="true">
                    ,
                   </mo>
    <mi>
                    c
                   </mi>
    </mrow>
    </msub>
    </mrow>
    <annotation encoding="application/x-tex">
                 ell(x, y) = L = {l_1,dots,l_N}^top, quad
    l_n = - sum_{c=1}^C w_c log frac{exp(x_{n,c})}{sum_{i=1}^C exp(x_{n,i})} y_{n,c}
                </annotation>
    </semantics>
    </math> -->
    ℓ ( x , y ) = L = { l 1 , … , l N } ⊤ , l n = − ∑ c = 1 C w c log ⁡ exp ⁡ ( x n , c ) ∑ i = 1 C exp ⁡ ( x n , i ) y n , c ell(x, y) = L = {l_1,dots,l_N}^top, quad
    l_n = - sum_{c=1}^C w_c log frac{exp(x_{n,c})}{sum_{i=1}^C exp(x_{n,i})} y_{n,c}

    ℓ ( x , y ) = L = { l 1 ​ , … , l N ​ } ⊤ , l n ​ = − c = 1 ∑ C ​ w c ​ lo g ∑ i = 1 C ​ exp ( x n , i ​ ) exp ( x n , c ​ ) ​ y n , c ​

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
        </math> -->w ww  is the weight, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
        <semantics>
        <mrow>
        <mi>
                      C
                     </mi>
        </mrow>
        <annotation encoding="application/x-tex">
                     C
                    </annotation>
        </semantics>
        </math> -->C CC  is the number of classes, and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
        </math> -->N NN  spans the minibatch dimension as well as <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
        <semantics>
        <mrow>
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
                       k
                      </mi>
        </msub>
        </mrow>
        <annotation encoding="application/x-tex">
                     d_1, ..., d_k
                    </annotation>
        </semantics>
        </math> -->d 1 , . . . , d k d_1, ..., d_kd 1 ​ , ... , d k ​  for the *K* -dimensional case. If `reduction`  is not `'none'`  (default `'mean'`  ), then

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
        <mfrac>
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
        </mrow>
        <mi>
                             N
                            </mi>
        </mfrac>
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
            frac{sum_{n=1}^N l_n}{N}, &amp;
             text{if reduction} = text{`mean';}
              sum_{n=1}^N l_n,  &amp;
              text{if reduction} = text{`sum'.}
          end{cases}
                    </annotation>
        </semantics>
        </math> -->
        ℓ ( x , y ) = { ∑ n = 1 N l n N , if reduction = ‘mean’; ∑ n = 1 N l n , if reduction = ‘sum’. ell(x, y) = begin{cases}
         frac{sum_{n=1}^N l_n}{N}, &
         text{if reduction} = text{`mean';}
         sum_{n=1}^N l_n, &
         text{if reduction} = text{`sum'.}
         end{cases}

    ℓ ( x , y ) = { N ∑ n = 1 N ​ l n ​ ​ , ∑ n = 1 N ​ l n ​ , ​ if reduction = ‘mean’; if reduction = ‘sum’. ​

Note 

The performance of this criterion is generally better when *target* contains class
indices, as this allows for optimized computation. Consider providing *target* as
class probabilities only when a single class label per minibatch item is too restrictive.

Parameters
:   * **weight** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – a manual rescaling weight given to each class.
If given, has to be a Tensor of size *C* .
* **size_average** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ). By default,
the losses are averaged over each loss element in the batch. Note that for
some losses, there are multiple elements per sample. If the field `size_average`  is set to `False`  , the losses are instead summed for each minibatch. Ignored
when `reduce`  is `False`  . Default: `True`
* **ignore_index** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – Specifies a target value that is ignored
and does not contribute to the input gradient. When `size_average`  is `True`  , the loss is averaged over non-ignored targets. Note that `ignore_index`  is only applicable when the target contains class indices.
* **reduce** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ). By default, the
losses are averaged or summed over observations for each minibatch depending
on `size_average`  . When `reduce`  is `False`  , returns a loss per
batch element instead and ignores `size_average`  . Default: `True`
* **reduction** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – Specifies the reduction to apply to the output: `'none'`  | `'mean'`  | `'sum'`  . `'none'`  : no reduction will
be applied, `'mean'`  : the weighted mean of the output is taken, `'sum'`  : the output will be summed. Note: `size_average`  and `reduce`  are in the process of being deprecated, and in
the meantime, specifying either of those two args will override `reduction`  . Default: `'mean'`
* **label_smoothing** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – A float in [0.0, 1.0]. Specifies the amount
of smoothing when computing the loss, where 0.0 means no smoothing. The targets
become a mixture of the original ground truth and a uniform distribution as described in [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)  . Default: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
                0.0
               </mn>
</mrow>
<annotation encoding="application/x-tex">
               0.0
              </annotation>
</semantics>
</math> -->0.0 0.00.0  .

Shape:
:   * Input: Shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( C ) (C)( C )  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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

* Target: If containing class indices, shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( ) ()( )  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->K ≥ 1 K geq 1K ≥ 1  in the case of K-dimensional loss where each value should be between <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               [0, C)
              </annotation>
</semantics>
</math> -->[ 0 , C ) [0, C)[ 0 , C )  . The
target data type is required to be long when using class indices. If containing class probabilities, the
target must be the same shape input, and each value should be between <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mn>
                1
               </mn>
<mo stretchy="false">
                ]
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               [0, 1]
              </annotation>
</semantics>
</math> -->[ 0 , 1 ] [0, 1][ 0 , 1 ]  . This means the target
data type is required to be float when using class probabilities.

* Output: If reduction is ‘none’, shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( ) ()( )  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->K ≥ 1 K geq 1K ≥ 1  in the case of K-dimensional loss, depending on the shape of the input. Otherwise, scalar.

where: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right left" columnspacing="0em" rowspacing="0.25em">
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mi>
                  C
                 </mi>
<mo>
                  =
                 </mo>
<mrow>
</mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mtext>
                  number of classes
                 </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mi>
                  N
                 </mi>
<mo>
                  =
                 </mo>
<mrow>
</mrow>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mrow>
</mrow>
<mtext>
                  batch size
                 </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
             begin{aligned}
    C ={} &amp; text{number of classes} 
    N ={} &amp; text{batch size} 
end{aligned}
            </annotation>
</semantics>
</math> -->
C = number of classes N = batch size begin{aligned}
 C ={} & text{number of classes} 
 N ={} & text{batch size} 
end{aligned}

C = N = ​ number of classes batch size ​

Examples 

```
>>> # Example of target with class indices
>>> loss = nn.CrossEntropyLoss()
>>> input = torch.randn(3, 5, requires_grad=True)
>>> target = torch.empty(3, dtype=torch.long).random_(5)
>>> output = loss(input, target)
>>> output.backward()
>>>
>>> # Example of target with class probabilities
>>> input = torch.randn(3, 5, requires_grad=True)
>>> target = torch.randn(3, 5).softmax(dim=1)
>>> output = loss(input, target)
>>> output.backward()

```

