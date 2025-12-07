HuberLoss 
======================================================

*class* torch.nn. HuberLoss ( *reduction = 'mean'*  , *delta = 1.0* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py#L1055) 
:   Creates a criterion that uses a squared term if the absolute
element-wise error falls below delta and a delta-scaled L1 term otherwise.
This loss combines advantages of both [`L1Loss`](torch.nn.L1Loss.html#torch.nn.L1Loss "torch.nn.L1Loss")  and [`MSELoss`](torch.nn.MSELoss.html#torch.nn.MSELoss "torch.nn.MSELoss")  ; the
delta-scaled L1 region makes the loss less sensitive to outliers than [`MSELoss`](torch.nn.MSELoss.html#torch.nn.MSELoss "torch.nn.MSELoss")  ,
while the L2 region provides smoothness over [`L1Loss`](torch.nn.L1Loss.html#torch.nn.L1Loss "torch.nn.L1Loss")  near 0. See [Huber loss](https://en.wikipedia.org/wiki/Huber_loss)  for more information. 

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
                  d
                 </mi>
<mi>
                  e
                 </mi>
<mi>
                  l
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
<mi>
                  d
                 </mi>
<mi>
                  e
                 </mi>
<mi>
                  l
                 </mi>
<mi>
                  t
                 </mi>
<mi>
                  a
                 </mi>
<mo>
                  ∗
                 </mo>
<mo stretchy="false">
                  (
                 </mo>
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
                  d
                 </mi>
<mi>
                  e
                 </mi>
<mi>
                  l
                 </mi>
<mi>
                  t
                 </mi>
<mi>
                  a
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
0.5 (x_n - y_n)^2, &amp; text{if } |x_n - y_n| &lt; delta 
delta * (|x_n - y_n| - 0.5 * delta), &amp; text{otherwise }
end{cases}
          </annotation>
</semantics>
</math> -->
l n = { 0.5 ( x n − y n ) 2 , if ∣ x n − y n ∣ < d e l t a d e l t a ∗ ( ∣ x n − y n ∣ − 0.5 ∗ d e l t a ) , otherwise l_n = begin{cases}
0.5 (x_n - y_n)^2, & text{if } |x_n - y_n| < delta 
delta * (|x_n - y_n| - 0.5 * delta), & text{otherwise }
end{cases}

l n ​ = { 0.5 ( x n ​ − y n ​ ) 2 , d e lt a ∗ ( ∣ x n ​ − y n ​ ∣ − 0.5 ∗ d e lt a ) , ​ if ∣ x n ​ − y n ​ ∣ < d e lt a otherwise ​

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

When delta is set to 1, this loss is equivalent to [`SmoothL1Loss`](torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss "torch.nn.SmoothL1Loss")  .
In general, this loss differs from [`SmoothL1Loss`](torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss "torch.nn.SmoothL1Loss")  by a factor of delta (AKA beta
in Smooth L1).
See [`SmoothL1Loss`](torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss "torch.nn.SmoothL1Loss")  for additional discussion on the differences in behavior
between the two losses.

Parameters
:   * **reduction** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – Specifies the reduction to apply to the output: `'none'`  | `'mean'`  | `'sum'`  . `'none'`  : no reduction will be applied, `'mean'`  : the sum of the output will be divided by the number of
elements in the output, `'sum'`  : the output will be summed. Default: `'mean'`
* **delta** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – Specifies the threshold at which to change between delta-scaled L1 and L2 loss.
The value must be positive. Default: 1.0

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

