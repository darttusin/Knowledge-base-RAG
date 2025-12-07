GaussianNLLLoss 
==================================================================

*class* torch.nn. GaussianNLLLoss ( *** , *full = False*  , *eps = 1e-06*  , *reduction = 'mean'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py#L367) 
:   Gaussian negative log likelihood loss. 

The targets are treated as samples from Gaussian distributions with
expectations and variances predicted by the neural network. For a `target`  tensor modelled as having Gaussian distribution with a tensor
of expectations `input`  and a tensor of positive variances `var`  the loss is: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            loss
           </mtext>
<mo>
            =
           </mo>
<mfrac>
<mn>
             1
            </mn>
<mn>
             2
            </mn>
</mfrac>
<mrow>
<mo fence="true">
             (
            </mo>
<mi>
             log
            </mi>
<mo>
             ⁡
            </mo>
<mrow>
<mo fence="true">
              (
             </mo>
<mtext>
              max
             </mtext>
<mrow>
<mo fence="true">
               (
              </mo>
<mtext>
               var
              </mtext>
<mo separator="true">
               ,
              </mo>
<mtext>
               eps
              </mtext>
<mo fence="true">
               )
              </mo>
</mrow>
<mo fence="true">
              )
             </mo>
</mrow>
<mo>
             +
            </mo>
<mfrac>
<msup>
<mrow>
<mo fence="true">
                (
               </mo>
<mtext>
                input
               </mtext>
<mo>
                −
               </mo>
<mtext>
                target
               </mtext>
<mo fence="true">
                )
               </mo>
</mrow>
<mn>
               2
              </mn>
</msup>
<mrow>
<mtext>
               max
              </mtext>
<mrow>
<mo fence="true">
                (
               </mo>
<mtext>
                var
               </mtext>
<mo separator="true">
                ,
               </mo>
<mtext>
                eps
               </mtext>
<mo fence="true">
                )
               </mo>
</mrow>
</mrow>
</mfrac>
<mo fence="true">
             )
            </mo>
</mrow>
<mo>
            +
           </mo>
<mtext>
            const.
           </mtext>
</mrow>
<annotation encoding="application/x-tex">
           text{loss} = frac{1}{2}left(logleft(text{max}left(text{var},
 text{eps}right)right) + frac{left(text{input} - text{target}right)^2}
{text{max}left(text{var},  text{eps}right)}right) + text{const.}
          </annotation>
</semantics>
</math> -->
loss = 1 2 ( log ⁡ ( max ( var , eps ) ) + ( input − target ) 2 max ( var , eps ) ) + const. text{loss} = frac{1}{2}left(logleft(text{max}left(text{var},
 text{eps}right)right) + frac{left(text{input} - text{target}right)^2}
{text{max}left(text{var},  text{eps}right)}right) + text{const.}

loss = 2 1 ​ ( lo g ( max ( var , eps ) ) + max ( var , eps ) ( input − target ) 2 ​ ) + const.

where `eps`  is used for stability. By default, the constant term of
the loss function is omitted unless `full`  is `True`  . If `var`  is not the same
size as `input`  (due to a homoscedastic assumption), it must either have a final dimension
of 1 or have one fewer dimension (with all other sizes being the same) for correct broadcasting. 

Parameters
:   * **full** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – include the constant term in the loss
calculation. Default: `False`  .
* **eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – value used to clamp `var`  (see note below), for
stability. Default: 1e-6.
* **reduction** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – specifies the reduction to apply to the
output: `'none'`  | `'mean'`  | `'sum'`  . `'none'`  : no reduction
will be applied, `'mean'`  : the output is the average of all batch
member losses, `'sum'`  : the output is the sum of all batch member
losses. Default: `'mean'`  .

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
</math> -->( N , ∗ ) (N, *)( N , ∗ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->∗ *∗  means any number of additional
dimensions

* Target: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N , ∗ ) (N, *)( N , ∗ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( ∗ ) (*)( ∗ )  , same shape as the input, or same shape as the input
but with one dimension equal to 1 (to allow for broadcasting)

* Var: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N , ∗ ) (N, *)( N , ∗ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( ∗ ) (*)( ∗ )  , same shape as the input, or same shape as the input but
with one dimension equal to 1, or same shape as the input but with one fewer
dimension (to allow for broadcasting), or a scalar value

* Output: scalar if `reduction`  is `'mean'`  (default) or `'sum'`  . If `reduction`  is `'none'`  , then <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( N , ∗ ) (N, *)( N , ∗ )  , same
shape as the input

Examples 

```
>>> loss = nn.GaussianNLLLoss()
>>> input = torch.randn(5, 2, requires_grad=True)
>>> target = torch.randn(5, 2)
>>> var = torch.ones(5, 2, requires_grad=True)  # heteroscedastic
>>> output = loss(input, target, var)
>>> output.backward()

```

```
>>> loss = nn.GaussianNLLLoss()
>>> input = torch.randn(5, 2, requires_grad=True)
>>> target = torch.randn(5, 2)
>>> var = torch.ones(5, 1, requires_grad=True)  # homoscedastic
>>> output = loss(input, target, var)
>>> output.backward()

```

Note 

The clamping of `var`  is ignored with respect to autograd, and so the
gradients are unaffected by it.

Reference:
:   Nix, D. A. and Weigend, A. S., “Estimating the mean and variance of the
target probability distribution”, Proceedings of 1994 IEEE International
Conference on Neural Networks (ICNN’94), Orlando, FL, USA, 1994, pp. 55-60
vol.1, doi: 10.1109/ICNN.1994.374138.

