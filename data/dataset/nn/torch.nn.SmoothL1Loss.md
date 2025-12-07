SmoothL1Loss 
============================================================

*class* torch.nn. SmoothL1Loss ( *size_average = None*  , *reduce = None*  , *reduction = 'mean'*  , *beta = 1.0* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py#L971) 
:   Creates a criterion that uses a squared term if the absolute
element-wise error falls below beta and an L1 term otherwise.
It is less sensitive to outliers than [`torch.nn.MSELoss`](torch.nn.MSELoss.html#torch.nn.MSELoss "torch.nn.MSELoss")  and in some cases
prevents exploding gradients (e.g. see the paper [Fast R-CNN](https://arxiv.org/abs/1504.08083)  by Ross Girshick). 

For a batch of size <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->N NN  , the unreduced loss can be described as: 

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
<mi>
             T
            </mi>
</msup>
</mrow>
<annotation encoding="application/x-tex">
           ell(x, y) = L = {l_1, ..., l_N}^T
          </annotation>
</semantics>
</math> -->
ℓ ( x , y ) = L = { l 1 , . . . , l N } T ell(x, y) = L = {l_1, ..., l_N}^T

ℓ ( x , y ) = L = { l 1 ​ , ... , l N ​ } T

with 

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
<mn>
                  0.5
                 </mn>
<mo stretchy="false">
                  (
                 </mo>
<msub>
<mi>
                   x
                  </mi>
<mi>
                   n
                  </mi>
</msub>
<mo>
                  −
                 </mo>
<msub>
<mi>
                   y
                  </mi>
<mi>
                   n
                  </mi>
</msub>
<msup>
<mo stretchy="false">
                   )
                  </mo>
<mn>
                   2
                  </mn>
</msup>
<mi mathvariant="normal">
                  /
                 </mi>
<mi>
                  b
                 </mi>
<mi>
                  e
                 </mi>
<mi>
                  t
                 </mi>
<mi>
                  a
                 </mi>
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
<mi mathvariant="normal">
                  ∣
                 </mi>
<msub>
<mi>
                   x
                  </mi>
<mi>
                   n
                  </mi>
</msub>
<mo>
                  −
                 </mo>
<msub>
<mi>
                   y
                  </mi>
<mi>
                   n
                  </mi>
</msub>
<mi mathvariant="normal">
                  ∣
                 </mi>
<mo>
                  &lt;
                 </mo>
<mi>
                  b
                 </mi>
<mi>
                  e
                 </mi>
<mi>
                  t
                 </mi>
<mi>
                  a
                 </mi>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mi mathvariant="normal">
                  ∣
                 </mi>
<msub>
<mi>
                   x
                  </mi>
<mi>
                   n
                  </mi>
</msub>
<mo>
                  −
                 </mo>
<msub>
<mi>
                   y
                  </mi>
<mi>
                   n
                  </mi>
</msub>
<mi mathvariant="normal">
                  ∣
                 </mi>
<mo>
                  −
                 </mo>
<mn>
                  0.5
                 </mn>
<mo>
                  ∗
                 </mo>
<mi>
                  b
                 </mi>
<mi>
                  e
                 </mi>
<mi>
                  t
                 </mi>
<mi>
                  a
                 </mi>
<mo separator="true">
                  ,
                 </mo>
</mrow>
</mstyle>
</mtd>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mtext>
                 otherwise
                </mtext>
</mstyle>
</mtd>
</mtr>
</mtable>
</mrow>
</mrow>
<annotation encoding="application/x-tex">
           l_n = begin{cases}
0.5 (x_n - y_n)^2 / beta, &amp; text{if } |x_n - y_n| &lt; beta 
|x_n - y_n| - 0.5 * beta, &amp; text{otherwise }
end{cases}
          </annotation>
</semantics>
</math> -->
l n = { 0.5 ( x n − y n ) 2 / b e t a , if ∣ x n − y n ∣ < b e t a ∣ x n − y n ∣ − 0.5 ∗ b e t a , otherwise l_n = begin{cases}
0.5 (x_n - y_n)^2 / beta, & text{if } |x_n - y_n| < beta 
|x_n - y_n| - 0.5 * beta, & text{otherwise }
end{cases}

l n ​ = { 0.5 ( x n ​ − y n ​ ) 2 / b e t a , ∣ x n ​ − y n ​ ∣ − 0.5 ∗ b e t a , ​ if ∣ x n ​ − y n ​ ∣ < b e t a otherwise ​

If *reduction* is not *none* , then: 

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

Note 

Smooth L1 loss can be seen as exactly [`L1Loss`](torch.nn.L1Loss.html#torch.nn.L1Loss "torch.nn.L1Loss")  , but with the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
             ∣
            </mi>
<mi>
             x
            </mi>
<mo>
             −
            </mo>
<mi>
             y
            </mi>
<mi mathvariant="normal">
             ∣
            </mi>
<mo>
             &lt;
            </mo>
<mi>
             b
            </mi>
<mi>
             e
            </mi>
<mi>
             t
            </mi>
<mi>
             a
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            |x - y| &lt; beta
           </annotation>
</semantics>
</math> -->∣ x − y ∣ < b e t a |x - y| < beta∣ x − y ∣ < b e t a  portion replaced with a quadratic function such that its slope is 1 at <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
             ∣
            </mi>
<mi>
             x
            </mi>
<mo>
             −
            </mo>
<mi>
             y
            </mi>
<mi mathvariant="normal">
             ∣
            </mi>
<mo>
             =
            </mo>
<mi>
             b
            </mi>
<mi>
             e
            </mi>
<mi>
             t
            </mi>
<mi>
             a
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            |x - y| = beta
           </annotation>
</semantics>
</math> -->∣ x − y ∣ = b e t a |x - y| = beta∣ x − y ∣ = b e t a  .
The quadratic segment smooths the L1 loss near <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="normal">
             ∣
            </mi>
<mi>
             x
            </mi>
<mo>
             −
            </mo>
<mi>
             y
            </mi>
<mi mathvariant="normal">
             ∣
            </mi>
<mo>
             =
            </mo>
<mn>
             0
            </mn>
</mrow>
<annotation encoding="application/x-tex">
            |x - y| = 0
           </annotation>
</semantics>
</math> -->∣ x − y ∣ = 0 |x - y| = 0∣ x − y ∣ = 0  .

Note 

Smooth L1 loss is closely related to [`HuberLoss`](torch.nn.HuberLoss.html#torch.nn.HuberLoss "torch.nn.HuberLoss")  , being
equivalent to <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             h
            </mi>
<mi>
             u
            </mi>
<mi>
             b
            </mi>
<mi>
             e
            </mi>
<mi>
             r
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
<mi mathvariant="normal">
             /
            </mi>
<mi>
             b
            </mi>
<mi>
             e
            </mi>
<mi>
             t
            </mi>
<mi>
             a
            </mi>
</mrow>
<annotation encoding="application/x-tex">
            huber(x, y) / beta
           </annotation>
</semantics>
</math> -->h u b e r ( x , y ) / b e t a huber(x, y) / betah u b er ( x , y ) / b e t a  (note that Smooth L1’s beta hyper-parameter is
also known as delta for Huber). This leads to the following differences: 

* As beta -> 0, Smooth L1 loss converges to [`L1Loss`](torch.nn.L1Loss.html#torch.nn.L1Loss "torch.nn.L1Loss")  , while [`HuberLoss`](torch.nn.HuberLoss.html#torch.nn.HuberLoss "torch.nn.HuberLoss")  converges to a constant 0 loss. When beta is 0, Smooth L1 loss is equivalent to L1 loss.
* As beta -> <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
               +
              </mo>
<mi mathvariant="normal">
               ∞
              </mi>
</mrow>
<annotation encoding="application/x-tex">
              +infty
             </annotation>
</semantics>
</math> -->+ ∞ +infty+ ∞  , Smooth L1 loss converges to a constant 0 loss, while [`HuberLoss`](torch.nn.HuberLoss.html#torch.nn.HuberLoss "torch.nn.HuberLoss")  converges to [`MSELoss`](torch.nn.MSELoss.html#torch.nn.MSELoss "torch.nn.MSELoss")  .

* For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant slope of 1.
For [`HuberLoss`](torch.nn.HuberLoss.html#torch.nn.HuberLoss "torch.nn.HuberLoss")  , the slope of the L1 segment is beta.

Parameters
:   * **size_average** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ). By default,
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
* **beta** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – Specifies the threshold at which to change between L1 and L2 loss.
The value must be non-negative. Default: 1.0

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
</math> -->( ∗ ) (*)( ∗ )  , where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->∗ *∗  means any number of dimensions.

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
</math> -->( ∗ ) (*)( ∗ )  , same shape as the input.

* Output: scalar. If `reduction`  is `'none'`  , then <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( ∗ ) (*)( ∗ )  , same shape as the input.

