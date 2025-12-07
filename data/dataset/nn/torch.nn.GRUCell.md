GRUCell 
==================================================

*class* torch.nn. GRUCell ( *input_size*  , *hidden_size*  , *bias = True*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/rnn.py#L1718) 
:   A gated recurrent unit (GRU) cell. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="left left" columnspacing="1em" rowspacing="0.16em">
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mi>
                r
               </mi>
<mo>
                =
               </mo>
<mi>
                σ
               </mi>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 W
                </mi>
<mrow>
<mi>
                  i
                 </mi>
<mi>
                  r
                 </mi>
</mrow>
</msub>
<mi>
                x
               </mi>
<mo>
                +
               </mo>
<msub>
<mi>
                 b
                </mi>
<mrow>
<mi>
                  i
                 </mi>
<mi>
                  r
                 </mi>
</mrow>
</msub>
<mo>
                +
               </mo>
<msub>
<mi>
                 W
                </mi>
<mrow>
<mi>
                  h
                 </mi>
<mi>
                  r
                 </mi>
</mrow>
</msub>
<mi>
                h
               </mi>
<mo>
                +
               </mo>
<msub>
<mi>
                 b
                </mi>
<mrow>
<mi>
                  h
                 </mi>
<mi>
                  r
                 </mi>
</mrow>
</msub>
<mo stretchy="false">
                )
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
                z
               </mi>
<mo>
                =
               </mo>
<mi>
                σ
               </mi>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 W
                </mi>
<mrow>
<mi>
                  i
                 </mi>
<mi>
                  z
                 </mi>
</mrow>
</msub>
<mi>
                x
               </mi>
<mo>
                +
               </mo>
<msub>
<mi>
                 b
                </mi>
<mrow>
<mi>
                  i
                 </mi>
<mi>
                  z
                 </mi>
</mrow>
</msub>
<mo>
                +
               </mo>
<msub>
<mi>
                 W
                </mi>
<mrow>
<mi>
                  h
                 </mi>
<mi>
                  z
                 </mi>
</mrow>
</msub>
<mi>
                h
               </mi>
<mo>
                +
               </mo>
<msub>
<mi>
                 b
                </mi>
<mrow>
<mi>
                  h
                 </mi>
<mi>
                  z
                 </mi>
</mrow>
</msub>
<mo stretchy="false">
                )
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
                n
               </mi>
<mo>
                =
               </mo>
<mi>
                tanh
               </mi>
<mo>
                ⁡
               </mo>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 W
                </mi>
<mrow>
<mi>
                  i
                 </mi>
<mi>
                  n
                 </mi>
</mrow>
</msub>
<mi>
                x
               </mi>
<mo>
                +
               </mo>
<msub>
<mi>
                 b
                </mi>
<mrow>
<mi>
                  i
                 </mi>
<mi>
                  n
                 </mi>
</mrow>
</msub>
<mo>
                +
               </mo>
<mi>
                r
               </mi>
<mo>
                ⊙
               </mo>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 W
                </mi>
<mrow>
<mi>
                  h
                 </mi>
<mi>
                  n
                 </mi>
</mrow>
</msub>
<mi>
                h
               </mi>
<mo>
                +
               </mo>
<msub>
<mi>
                 b
                </mi>
<mrow>
<mi>
                  h
                 </mi>
<mi>
                  n
                 </mi>
</mrow>
</msub>
<mo stretchy="false">
                )
               </mo>
<mo stretchy="false">
                )
               </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<msup>
<mi>
                 h
                </mi>
<mo lspace="0em" mathvariant="normal" rspace="0em">
                 ′
                </mo>
</msup>
<mo>
                =
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
                z
               </mi>
<mo stretchy="false">
                )
               </mo>
<mo>
                ⊙
               </mo>
<mi>
                n
               </mi>
<mo>
                +
               </mo>
<mi>
                z
               </mi>
<mo>
                ⊙
               </mo>
<mi>
                h
               </mi>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
           begin{array}{ll}
r = sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) 
z = sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) 
n = tanh(W_{in} x + b_{in} + r odot (W_{hn} h + b_{hn})) 
h' = (1 - z) odot n + z odot h
end{array}
          </annotation>
</semantics>
</math> -->
r = σ ( W i r x + b i r + W h r h + b h r ) z = σ ( W i z x + b i z + W h z h + b h z ) n = tanh ⁡ ( W i n x + b i n + r ⊙ ( W h n h + b h n ) ) h ′ = ( 1 − z ) ⊙ n + z ⊙ h begin{array}{ll}
r = sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) 
z = sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) 
n = tanh(W_{in} x + b_{in} + r odot (W_{hn} h + b_{hn})) 
h' = (1 - z) odot n + z odot h
end{array}

r = σ ( W i r ​ x + b i r ​ + W h r ​ h + b h r ​ ) z = σ ( W i z ​ x + b i z ​ + W h z ​ h + b h z ​ ) n = tanh ( W in ​ x + b in ​ + r ⊙ ( W hn ​ h + b hn ​ )) h ′ = ( 1 − z ) ⊙ n + z ⊙ h ​

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            σ
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           sigma
          </annotation>
</semantics>
</math> -->σ sigmaσ  is the sigmoid function, and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo>
            ⊙
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           odot
          </annotation>
</semantics>
</math> -->⊙ odot⊙  is the Hadamard product. 

Parameters
:   * **input_size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – The number of expected features in the input *x*
* **hidden_size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – The number of features in the hidden state *h*
* **bias** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If `False`  , then the layer does not use bias weights *b_ih* and *b_hh* . Default: `True`

Inputs: input, hidden
:   * **input** : tensor containing input features
* **hidden** : tensor containing the initial hidden
state for each element in the batch.
Defaults to zero if not provided.

Outputs: h’
:   * **h’** : tensor containing the next hidden state
for each element in the batch

Shape:
:   * input: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                 H
                </mi>
<mrow>
<mi>
                  i
                 </mi>
<mi>
                  n
                 </mi>
</mrow>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, H_{in})
              </annotation>
</semantics>
</math> -->( N , H i n ) (N, H_{in})( N , H in ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 H
                </mi>
<mrow>
<mi>
                  i
                 </mi>
<mi>
                  n
                 </mi>
</mrow>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (H_{in})
              </annotation>
</semantics>
</math> -->( H i n ) (H_{in})( H in ​ )  tensor containing input features where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
                 H
                </mi>
<mrow>
<mi>
                  i
                 </mi>
<mi>
                  n
                 </mi>
</mrow>
</msub>
</mrow>
<annotation encoding="application/x-tex">
               H_{in}
              </annotation>
</semantics>
</math> -->H i n H_{in}H in ​  = *input_size* .

* hidden: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                 H
                </mi>
<mrow>
<mi>
                  o
                 </mi>
<mi>
                  u
                 </mi>
<mi>
                  t
                 </mi>
</mrow>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, H_{out})
              </annotation>
</semantics>
</math> -->( N , H o u t ) (N, H_{out})( N , H o u t ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 H
                </mi>
<mrow>
<mi>
                  o
                 </mi>
<mi>
                  u
                 </mi>
<mi>
                  t
                 </mi>
</mrow>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (H_{out})
              </annotation>
</semantics>
</math> -->( H o u t ) (H_{out})( H o u t ​ )  tensor containing the initial hidden
state where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
                 H
                </mi>
<mrow>
<mi>
                  o
                 </mi>
<mi>
                  u
                 </mi>
<mi>
                  t
                 </mi>
</mrow>
</msub>
</mrow>
<annotation encoding="application/x-tex">
               H_{out}
              </annotation>
</semantics>
</math> -->H o u t H_{out}H o u t ​  = *hidden_size* . Defaults to zero if not provided.

* output: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                 H
                </mi>
<mrow>
<mi>
                  o
                 </mi>
<mi>
                  u
                 </mi>
<mi>
                  t
                 </mi>
</mrow>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (N, H_{out})
              </annotation>
</semantics>
</math> -->( N , H o u t ) (N, H_{out})( N , H o u t ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<msub>
<mi>
                 H
                </mi>
<mrow>
<mi>
                  o
                 </mi>
<mi>
                  u
                 </mi>
<mi>
                  t
                 </mi>
</mrow>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (H_{out})
              </annotation>
</semantics>
</math> -->( H o u t ) (H_{out})( H o u t ​ )  tensor containing the next hidden state.

Variables
:   * **weight_ih** ( [*torch.Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the learnable input-hidden weights, of shape *(3*hidden_size, input_size)*
* **weight_hh** ( [*torch.Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the learnable hidden-hidden weights, of shape *(3*hidden_size, hidden_size)*
* **bias_ih** – the learnable input-hidden bias, of shape *(3*hidden_size)*
* **bias_hh** – the learnable hidden-hidden bias, of shape *(3*hidden_size)*

Note 

All the weights and biases are initialized from <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi mathvariant="script">
             U
            </mi>
<mo stretchy="false">
             (
            </mo>
<mo>
             −
            </mo>
<msqrt>
<mi>
              k
             </mi>
</msqrt>
<mo separator="true">
             ,
            </mo>
<msqrt>
<mi>
              k
             </mi>
</msqrt>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            mathcal{U}(-sqrt{k}, sqrt{k})
           </annotation>
</semantics>
</math> -->U ( − k , k ) mathcal{U}(-sqrt{k}, sqrt{k})U ( − k ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ , k ![SVG Image](data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjEuMDhlbSIgcHJlc2VydmVhc3BlY3RyYXRpbz0ieE1pbllNaW4gc2xpY2UiIHZpZXdib3g9IjAgMCA0MDAwMDAgMTA4MCIgd2lkdGg9IjQwMGVtIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNOTUsNzAyCmMtMi43LDAsLTcuMTcsLTIuNywtMTMuNSwtOGMtNS44LC01LjMsLTkuNSwtMTAsLTkuNSwtMTQKYzAsLTIsMC4zLC0zLjMsMSwtNGMxLjMsLTIuNywyMy44MywtMjAuNyw2Ny41LC01NApjNDQuMiwtMzMuMyw2NS44LC01MC4zLDY2LjUsLTUxYzEuMywtMS4zLDMsLTIsNSwtMmM0LjcsMCw4LjcsMy4zLDEyLDEwCnMxNzMsMzc4LDE3MywzNzhjMC43LDAsMzUuMywtNzEsMTA0LC0yMTNjNjguNywtMTQyLDEzNy41LC0yODUsMjA2LjUsLTQyOQpjNjksLTE0NCwxMDQuNSwtMjE3LjcsMTA2LjUsLTIyMQpsMCAtMApjNS4zLC05LjMsMTIsLTE0LDIwLC0xNApINDAwMDAwdjQwSDg0NS4yNzI0CnMtMjI1LjI3Miw0NjcsLTIyNS4yNzIsNDY3cy0yMzUsNDg2LC0yMzUsNDg2Yy0yLjcsNC43LC05LDcsLTE5LDcKYy02LDAsLTEwLC0xLC0xMiwtM3MtMTk0LC00MjIsLTE5NCwtNDIycy02NSw0NywtNjUsNDd6Ck04MzQgODBoNDAwMDAwdjQwaC00MDAwMDB6Ij4KPC9wYXRoPgo8L3N2Zz4=)​ )  where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
             k
            </mi>
<mo>
             =
            </mo>
<mfrac>
<mn>
              1
             </mn>
<mtext>
              hidden_size
             </mtext>
</mfrac>
</mrow>
<annotation encoding="application/x-tex">
            k = frac{1}{text{hidden_size}}
           </annotation>
</semantics>
</math> -->k = 1 hidden_size k = frac{1}{text{hidden_size}}k = hidden_size 1 ​

On certain ROCm devices, when using float16 inputs this module will use [different precision](../notes/numerical_accuracy.html#fp16-on-mi200)  for backward. 

Examples: 

```
>>> rnn = nn.GRUCell(10, 20)
>>> input = torch.randn(6, 3, 10)
>>> hx = torch.randn(3, 20)
>>> output = []
>>> for i in range(6):
...     hx = rnn(input[i], hx)
...     output.append(hx)

```

