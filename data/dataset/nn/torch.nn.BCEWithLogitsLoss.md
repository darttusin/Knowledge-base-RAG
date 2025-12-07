BCEWithLogitsLoss 
======================================================================

*class* torch.nn. BCEWithLogitsLoss ( *weight = None*  , *size_average = None*  , *reduce = None*  , *reduction = 'mean'*  , *pos_weight = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py#L711) 
:   This loss combines a *Sigmoid* layer and the *BCELoss* in one single
class. This version is more numerically stable than using a plain *Sigmoid* followed by a *BCELoss* as, by combining the operations into one layer,
we take advantage of the log-sum-exp trick for numerical stability. 

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
<mi>
             σ
            </mi>
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
<mo stretchy="false">
             )
            </mo>
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
<mi>
             σ
            </mi>
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
<mo stretchy="false">
             )
            </mo>
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
l_n = - w_n left[ y_n cdot log sigma(x_n)
+ (1 - y_n) cdot log (1 - sigma(x_n)) right],
          </annotation>
</semantics>
</math> -->
ℓ ( x , y ) = L = { l 1 , … , l N } ⊤ , l n = − w n [ y n ⋅ log ⁡ σ ( x n ) + ( 1 − y n ) ⋅ log ⁡ ( 1 − σ ( x n ) ) ] , ell(x, y) = L = {l_1,dots,l_N}^top, quad
l_n = - w_n left[ y_n cdot log sigma(x_n)
+ (1 - y_n) cdot log (1 - sigma(x_n)) right],

ℓ ( x , y ) = L = { l 1 ​ , … , l N ​ } ⊤ , l n ​ = − w n ​ [ y n ​ ⋅ lo g σ ( x n ​ ) + ( 1 − y n ​ ) ⋅ lo g ( 1 − σ ( x n ​ )) ] ,

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
an auto-encoder. Note that the targets *t[i]* should be numbers
between 0 and 1. 

It’s possible to trade off recall and precision by adding weights to positive examples.
In the case of multi-label classification the loss can be described as: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi mathvariant="normal">
             ℓ
            </mi>
<mi>
             c
            </mi>
</msub>
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
<msub>
<mi>
             L
            </mi>
<mi>
             c
            </mi>
</msub>
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
<mrow>
<mn>
              1
             </mn>
<mo separator="true">
              ,
             </mo>
<mi>
              c
             </mi>
</mrow>
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
<mrow>
<mi>
              N
             </mi>
<mo separator="true">
              ,
             </mo>
<mi>
              c
             </mi>
</mrow>
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
<mrow>
<mo fence="true">
             [
            </mo>
<msub>
<mi>
              p
             </mi>
<mi>
              c
             </mi>
</msub>
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
<mo>
             ⋅
            </mo>
<mi>
             log
            </mi>
<mo>
             ⁡
            </mo>
<mi>
             σ
            </mi>
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
<mi>
             σ
            </mi>
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
           ell_c(x, y) = L_c = {l_{1,c},dots,l_{N,c}}^top, quad
l_{n,c} = - w_{n,c} left[ p_c y_{n,c} cdot log sigma(x_{n,c})
+ (1 - y_{n,c}) cdot log (1 - sigma(x_{n,c})) right],
          </annotation>
</semantics>
</math> -->
ℓ c ( x , y ) = L c = { l 1 , c , … , l N , c } ⊤ , l n , c = − w n , c [ p c y n , c ⋅ log ⁡ σ ( x n , c ) + ( 1 − y n , c ) ⋅ log ⁡ ( 1 − σ ( x n , c ) ) ] , ell_c(x, y) = L_c = {l_{1,c},dots,l_{N,c}}^top, quad
l_{n,c} = - w_{n,c} left[ p_c y_{n,c} cdot log sigma(x_{n,c})
+ (1 - y_{n,c}) cdot log (1 - sigma(x_{n,c})) right],

ℓ c ​ ( x , y ) = L c ​ = { l 1 , c ​ , … , l N , c ​ } ⊤ , l n , c ​ = − w n , c ​ [ p c ​ y n , c ​ ⋅ lo g σ ( x n , c ​ ) + ( 1 − y n , c ​ ) ⋅ lo g ( 1 − σ ( x n , c ​ )) ] ,

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            c
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           c
          </annotation>
</semantics>
</math> -->c cc  is the class number ( <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            c
           </mi>
<mo>
            &gt;
           </mo>
<mn>
            1
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           c &gt; 1
          </annotation>
</semantics>
</math> -->c > 1 c > 1c > 1  for multi-label binary classification, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
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
<annotation encoding="application/x-tex">
           c = 1
          </annotation>
</semantics>
</math> -->c = 1 c = 1c = 1  for single-label binary classification), <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->n nn  is the number of the sample in the batch and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             p
            </mi>
<mi>
             c
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           p_c
          </annotation>
</semantics>
</math> -->p c p_cp c ​  is the weight of the positive answer for the class <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            c
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           c
          </annotation>
</semantics>
</math> -->c cc  . 

<!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             p
            </mi>
<mi>
             c
            </mi>
</msub>
<mo>
            &gt;
           </mo>
<mn>
            1
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           p_c &gt; 1
          </annotation>
</semantics>
</math> -->p c > 1 p_c > 1p c ​ > 1  increases the recall, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             p
            </mi>
<mi>
             c
            </mi>
</msub>
<mo>
            &lt;
           </mo>
<mn>
            1
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           p_c &lt; 1
          </annotation>
</semantics>
</math> -->p c < 1 p_c < 1p c ​ < 1  increases the precision. 

For example, if a dataset contains 100 positive and 300 negative examples of a single class,
then `pos_weight`  for the class should be equal to <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mfrac>
<mn>
             300
            </mn>
<mn>
             100
            </mn>
</mfrac>
<mo>
            =
           </mo>
<mn>
            3
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           frac{300}{100}=3
          </annotation>
</semantics>
</math> -->300 100 = 3 frac{300}{100}=3100 300 ​ = 3  .
The loss would act as if the dataset contains <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
            3
           </mn>
<mo>
            ×
           </mo>
<mn>
            100
           </mn>
<mo>
            =
           </mo>
<mn>
            300
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           3times 100=300
          </annotation>
</semantics>
</math> -->3 × 100 = 300 3times 100=3003 × 100 = 300  positive examples. 

Examples 

```
>>> target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
>>> output = torch.full([10, 64], 1.5)  # A prediction (logit)
>>> pos_weight = torch.ones([64])  # All weights are equal to 1
>>> criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
>>> criterion(output, target)  # -log(sigmoid(1.5))
tensor(0.20...)

```

In the above example, the `pos_weight`  tensor’s elements correspond to the 64 distinct classes
in a multi-label binary classification scenario. Each element in `pos_weight`  is designed to adjust the
loss function based on the imbalance between negative and positive samples for the respective class.
This approach is useful in datasets with varying levels of class imbalance, ensuring that the loss
calculation accurately accounts for the distribution in each class. 

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
* **pos_weight** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – a weight of positive examples to be broadcasted with target.
Must be a tensor with equal size along the class dimension to the number of classes.
Pay close attention to PyTorch’s broadcasting semantics in order to achieve the desired
operations. For a target of size [B, C, H, W] (where B is batch size) pos_weight of
size [B, C, H, W] will apply different pos_weights to each element of the batch or
[C, H, W] the same pos_weights across the batch. To apply the same positive weight
along all spacial dimensions for a 2D multi-class target [C, H, W] use: [C, 1, 1].
Default: `None`

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
>>> loss = nn.BCEWithLogitsLoss()
>>> input = torch.randn(3, requires_grad=True)
>>> target = torch.empty(3).random_(2)
>>> output = loss(input, target)
>>> output.backward()

```

