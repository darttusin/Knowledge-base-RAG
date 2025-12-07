BCELoss 
==================================================

*class* torch.nn. BCELoss ( *weight = None*  , *size_average = None*  , *reduce = None*  , *reduction = 'mean'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py#L619) 
:   Creates a criterion that measures the Binary Cross Entropy between the target and
the input probabilities: 

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
<mi>
             n
            </mi>
</msub>
<mrow>
<mo fence="true">
             [
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
             ⋅
            </mo>
<mi>
             log
            </mi>
<mo>
             ⁡
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
             +
            </mo>
<mo stretchy="false">
             (
            </mo>
<mn>
             1
            </mn>
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
<mo stretchy="false">
             )
            </mo>
<mo>
             ⋅
            </mo>
<mi>
             log
            </mi>
<mo>
             ⁡
            </mo>
<mo stretchy="false">
             (
            </mo>
<mn>
             1
            </mn>
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
             )
            </mo>
<mo fence="true">
             ]
            </mo>
</mrow>
<mo separator="true">
            ,
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           ell(x, y) = L = {l_1,dots,l_N}^top, quad
l_n = - w_n left[ y_n cdot log x_n + (1 - y_n) cdot log (1 - x_n) right],
          </annotation>
</semantics>
</math> -->
ℓ ( x , y ) = L = { l 1 , … , l N } ⊤ , l n = − w n [ y n ⋅ log ⁡ x n + ( 1 − y n ) ⋅ log ⁡ ( 1 − x n ) ] , ell(x, y) = L = {l_1,dots,l_N}^top, quad
l_n = - w_n left[ y_n cdot log x_n + (1 - y_n) cdot log (1 - x_n) right],

ℓ ( x , y ) = L = { l 1 ​ , … , l N ​ } ⊤ , l n ​ = − w n ​ [ y n ​ ⋅ lo g x n ​ + ( 1 − y n ​ ) ⋅ lo g ( 1 − x n ​ ) ] ,

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

This is used for measuring the error of a reconstruction in for example
an auto-encoder. Note that the targets <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->y yy  should be numbers
between 0 and 1. 

Notice that if <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             x
            </mi>
<mi>
             n
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           x_n
          </annotation>
</semantics>
</math> -->x n x_nx n ​  is either 0 or 1, one of the log terms would be
mathematically undefined in the above loss equation. PyTorch chooses to set <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            log
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
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mo>
            −
           </mo>
<mi mathvariant="normal">
            ∞
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           log (0) = -infty
          </annotation>
</semantics>
</math> -->log ⁡ ( 0 ) = − ∞ log (0) = -inftylo g ( 0 ) = − ∞  , since <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mrow>
<mi>
              lim
             </mi>
<mo>
              ⁡
             </mo>
</mrow>
<mrow>
<mi>
              x
             </mi>
<mo>
              →
             </mo>
<mn>
              0
             </mn>
</mrow>
</msub>
<mi>
            log
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mo>
            −
           </mo>
<mi mathvariant="normal">
            ∞
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           lim_{xto 0} log (x) = -infty
          </annotation>
</semantics>
</math> -->lim ⁡ x → 0 log ⁡ ( x ) = − ∞ lim_{xto 0} log (x) = -inftylim x → 0 ​ lo g ( x ) = − ∞  .
However, an infinite term in the loss equation is not desirable for several reasons. 

For one, if either <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
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
            0
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           y_n = 0
          </annotation>
</semantics>
</math> -->y n = 0 y_n = 0y n ​ = 0  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
            (
           </mo>
<mn>
            1
           </mn>
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
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mn>
            0
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           (1 - y_n) = 0
          </annotation>
</semantics>
</math> -->( 1 − y n ) = 0 (1 - y_n) = 0( 1 − y n ​ ) = 0  , then we would be
multiplying 0 with infinity. Secondly, if we have an infinite loss value, then
we would also have an infinite term in our gradient, since <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mrow>
<mi>
              lim
             </mi>
<mo>
              ⁡
             </mo>
</mrow>
<mrow>
<mi>
              x
             </mi>
<mo>
              →
             </mo>
<mn>
              0
             </mn>
</mrow>
</msub>
<mfrac>
<mi>
             d
            </mi>
<mrow>
<mi>
              d
             </mi>
<mi>
              x
             </mi>
</mrow>
</mfrac>
<mi>
            log
           </mi>
<mo>
            ⁡
           </mo>
<mo stretchy="false">
            (
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            =
           </mo>
<mi mathvariant="normal">
            ∞
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           lim_{xto 0} frac{d}{dx} log (x) = infty
          </annotation>
</semantics>
</math> -->lim ⁡ x → 0 d d x log ⁡ ( x ) = ∞ lim_{xto 0} frac{d}{dx} log (x) = inftylim x → 0 ​ d x d ​ lo g ( x ) = ∞  .
This would make BCELoss’s backward method nonlinear with respect to <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             x
            </mi>
<mi>
             n
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           x_n
          </annotation>
</semantics>
</math> -->x n x_nx n ​  ,
and using it for things like linear regression would not be straight-forward. 

Our solution is that BCELoss clamps its log function outputs to be greater than
or equal to -100. This way, we can always have a finite loss value and a linear
backward method. 

Parameters
:   * **weight** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – a manual rescaling weight given to the loss
of each batch element. If given, has to be a Tensor of size *nbatch* .
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
</math> -->( ∗ ) (*)( ∗ )  , same
shape as input.

Examples 

```
>>> m = nn.Sigmoid()
>>> loss = nn.BCELoss()
>>> input = torch.randn(3, 2, requires_grad=True)
>>> target = torch.rand(3, 2, requires_grad=False)
>>> output = loss(m(input), target)
>>> output.backward()

```

