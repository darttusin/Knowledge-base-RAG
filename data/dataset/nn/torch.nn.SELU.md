SELU 
============================================

*class* torch.nn. SELU ( *inplace = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py#L615) 
:   Applies the SELU function element-wise. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            SELU
           </mtext>
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
<mtext>
            scale
           </mtext>
<mo>
            ∗
           </mo>
<mo stretchy="false">
            (
           </mo>
<mi>
            max
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
<mo separator="true">
            ,
           </mo>
<mi>
            x
           </mi>
<mo stretchy="false">
            )
           </mo>
<mo>
            +
           </mo>
<mi>
            min
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
<mo separator="true">
            ,
           </mo>
<mi>
            α
           </mi>
<mo>
            ∗
           </mo>
<mo stretchy="false">
            (
           </mo>
<mi>
            exp
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
            −
           </mo>
<mn>
            1
           </mn>
<mo stretchy="false">
            )
           </mo>
<mo stretchy="false">
            )
           </mo>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           text{SELU}(x) = text{scale} * (max(0,x) + min(0, alpha * (exp(x) - 1)))
          </annotation>
</semantics>
</math> -->
SELU ( x ) = scale ∗ ( max ⁡ ( 0 , x ) + min ⁡ ( 0 , α ∗ ( exp ⁡ ( x ) − 1 ) ) ) text{SELU}(x) = text{scale} * (max(0,x) + min(0, alpha * (exp(x) - 1)))

SELU ( x ) = scale ∗ ( max ( 0 , x ) + min ( 0 , α ∗ ( exp ( x ) − 1 )))

with <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            α
           </mi>
<mo>
            =
           </mo>
<mn>
            1.6732632423543772848170429916717
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           alpha = 1.6732632423543772848170429916717
          </annotation>
</semantics>
</math> -->α = 1.6732632423543772848170429916717 alpha = 1.6732632423543772848170429916717α = 1.6732632423543772848170429916717  and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            scale
           </mtext>
<mo>
            =
           </mo>
<mn>
            1.0507009873554804934193349852946
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           text{scale} = 1.0507009873554804934193349852946
          </annotation>
</semantics>
</math> -->scale = 1.0507009873554804934193349852946 text{scale} = 1.0507009873554804934193349852946scale = 1.0507009873554804934193349852946  . 

Warning 

When using `kaiming_normal`  or `kaiming_normal_`  for initialisation, `nonlinearity='linear'`  should be used instead of `nonlinearity='selu'`  in order to get [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)  .
See [`torch.nn.init.calculate_gain()`](../nn.init.html#torch.nn.init.calculate_gain "torch.nn.init.calculate_gain")  for more information.

More details can be found in the paper [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)  . 

Parameters
: **inplace** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – can optionally do the operation in-place. Default: `False`

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

* Output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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

![../_images/SELU.png](../_images/SELU.png)

Examples: 

```
>>> m = nn.SELU()
>>> input = torch.randn(2)
>>> output = m(input)

```

