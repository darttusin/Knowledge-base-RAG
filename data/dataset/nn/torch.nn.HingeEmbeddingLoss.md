HingeEmbeddingLoss 
========================================================================

*class* torch.nn. HingeEmbeddingLoss ( *margin = 1.0*  , *size_average = None*  , *reduce = None*  , *reduction = 'mean'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py#L837) 
:   Measures the loss given an input tensor <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->x xx  and a labels tensor <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->y yy  (containing 1 or -1).
This is usually used for measuring whether two inputs are similar or
dissimilar, e.g. using the L1 pairwise distance as <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->x xx  , and is typically
used for learning nonlinear embeddings or semi-supervised learning. 

The loss function for <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->n nn  -th sample in the mini-batch is 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
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
<mrow>
<mo fence="true">
             {
            </mo>
<mtable columnalign="left left" columnspacing="1em" rowspacing="0.36em">
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<msub>
<mi>
                   x
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
                  if
                 </mtext>
<mtext>
</mtext>
<msub>
<mi>
                   y
                  </mi>
<mi>
                   n
                  </mi>
</msub>
<mo>
                  =
                 </mo>
<mn>
                  1
                 </mn>
<mo separator="true">
                  ,
                 </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mi>
                  max
                 </mi>
<mo>
                  ⁡
                 </mo>
<mo stretchy="false">
                  {
                 </mo>
<mn>
                  0
                 </mn>
<mo separator="true">
                  ,
                 </mo>
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
<mo>
                  −
                 </mo>
<msub>
<mi>
                   x
                  </mi>
<mi>
                   n
                  </mi>
</msub>
<mo stretchy="false">
                  }
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
                  if
                 </mtext>
<mtext>
</mtext>
<msub>
<mi>
                   y
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
<mn>
                  1
                 </mn>
<mo separator="true">
                  ,
                 </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           l_n = begin{cases}
    x_n, &amp; text{if}; y_n = 1,
    max {0, margin - x_n}, &amp; text{if}; y_n = -1,
end{cases}
          </annotation>
</semantics>
</math> -->
l n = { x n , if y n = 1 , max ⁡ { 0 , m a r g i n − x n } , if y n = − 1 , l_n = begin{cases}
 x_n, & text{if}; y_n = 1,
 max {0, margin - x_n}, & text{if}; y_n = -1,
end{cases}

l n ​ = { x n ​ , max { 0 , ma r g in − x n ​ } , ​ if y n ​ = 1 , if y n ​ = − 1 , ​

and the total loss functions is 

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
           ell(x, y) = begin{cases}
    operatorname{mean}(L), &amp; text{if reduction} = text{`mean';}
    operatorname{sum}(L),  &amp; text{if reduction} = text{`sum'.}
end{cases}
          </annotation>
</semantics>
</math> -->
ℓ ( x , y ) = { mean ⁡ ( L ) , if reduction = ‘mean’; sum ⁡ ( L ) , if reduction = ‘sum’. ell(x, y) = begin{cases}
 operatorname{mean}(L), & text{if reduction} = text{`mean';}
 operatorname{sum}(L), & text{if reduction} = text{`sum'.}
end{cases}

ℓ ( x , y ) = { mean ( L ) , sum ( L ) , ​ if reduction = ‘mean’; if reduction = ‘sum’. ​

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
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
</mrow>
<annotation encoding="application/x-tex">
           L = {l_1,dots,l_N}^top
          </annotation>
</semantics>
</math> -->L = { l 1 , … , l N } ⊤ L = {l_1,dots,l_N}^topL = { l 1 ​ , … , l N ​ } ⊤  . 

Parameters
:   * **margin** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – Has a default value of *1* .
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
<mo>
                ∗
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (*)
              </annotation>
</semantics>
</math> -->( ∗ ) (*)( ∗ )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->∗ *∗  means, any number of dimensions. The sum operation
operates over all the elements.

* Target: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mo>
                ∗
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (*)
              </annotation>
</semantics>
</math> -->( ∗ ) (*)( ∗ )  , same shape as the input

* Output: scalar. If `reduction`  is `'none'`  , then same shape as the input

