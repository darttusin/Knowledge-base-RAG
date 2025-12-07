LSTMCell 
====================================================

*class* torch.nn. LSTMCell ( *input_size*  , *hidden_size*  , *bias = True*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/rnn.py#L1608) 
:   A long short-term memory (LSTM) cell. 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="left left" columnspacing="1em" rowspacing="0.16em">
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<mi>
                i
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
                  i
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
                  i
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
                  i
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
                  i
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
                f
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
                  f
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
                  f
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
                  f
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
                  f
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
                g
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
                  g
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
                  g
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
                  g
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
                  g
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
                o
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
                  o
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
                  o
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
                  o
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
                  o
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
<msup>
<mi>
                 c
                </mi>
<mo lspace="0em" mathvariant="normal" rspace="0em">
                 ′
                </mo>
</msup>
<mo>
                =
               </mo>
<mi>
                f
               </mi>
<mo>
                ⊙
               </mo>
<mi>
                c
               </mi>
<mo>
                +
               </mo>
<mi>
                i
               </mi>
<mo>
                ⊙
               </mo>
<mi>
                g
               </mi>
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
<mi>
                o
               </mi>
<mo>
                ⊙
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
<msup>
<mi>
                 c
                </mi>
<mo lspace="0em" mathvariant="normal" rspace="0em">
                 ′
                </mo>
</msup>
<mo stretchy="false">
                )
               </mo>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
           begin{array}{ll}
i = sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) 
f = sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) 
g = tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) 
o = sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) 
c' = f odot c + i odot g 
h' = o odot tanh(c') 
end{array}
          </annotation>
</semantics>
</math> -->
i = σ ( W i i x + b i i + W h i h + b h i ) f = σ ( W i f x + b i f + W h f h + b h f ) g = tanh ⁡ ( W i g x + b i g + W h g h + b h g ) o = σ ( W i o x + b i o + W h o h + b h o ) c ′ = f ⊙ c + i ⊙ g h ′ = o ⊙ tanh ⁡ ( c ′ ) begin{array}{ll}
i = sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) 
f = sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) 
g = tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) 
o = sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) 
c' = f odot c + i odot g 
h' = o odot tanh(c') 
end{array}

i = σ ( W ii ​ x + b ii ​ + W hi ​ h + b hi ​ ) f = σ ( W i f ​ x + b i f ​ + W h f ​ h + b h f ​ ) g = tanh ( W i g ​ x + b i g ​ + W h g ​ h + b h g ​ ) o = σ ( W i o ​ x + b i o ​ + W h o ​ h + b h o ​ ) c ′ = f ⊙ c + i ⊙ g h ′ = o ⊙ tanh ( c ′ ) ​

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

Inputs: input, (h_0, c_0)
:   * **input** of shape *(batch, input_size)* or *(input_size)* : tensor containing input features
* **h_0** of shape *(batch, hidden_size)* or *(hidden_size)* : tensor containing the initial hidden state
* **c_0** of shape *(batch, hidden_size)* or *(hidden_size)* : tensor containing the initial cell state

    If *(h_0, c_0)* is not provided, both **h_0** and **c_0** default to zero.

Outputs: (h_1, c_1)
:   * **h_1** of shape *(batch, hidden_size)* or *(hidden_size)* : tensor containing the next hidden state
* **c_1** of shape *(batch, hidden_size)* or *(hidden_size)* : tensor containing the next cell state

Variables
:   * **weight_ih** ( [*torch.Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the learnable input-hidden weights, of shape *(4*hidden_size, input_size)*
* **weight_hh** ( [*torch.Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – the learnable hidden-hidden weights, of shape *(4*hidden_size, hidden_size)*
* **bias_ih** – the learnable input-hidden bias, of shape *(4*hidden_size)*
* **bias_hh** – the learnable hidden-hidden bias, of shape *(4*hidden_size)*

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
>>> rnn = nn.LSTMCell(10, 20)  # (input_size, hidden_size)
>>> input = torch.randn(2, 3, 10)  # (time_steps, batch, input_size)
>>> hx = torch.randn(3, 20)  # (batch, hidden_size)
>>> cx = torch.randn(3, 20)
>>> output = []
>>> for i in range(input.size()[0]):
...     hx, cx = rnn(input[i], (hx, cx))
...     output.append(hx)
>>> output = torch.stack(output, dim=0)

```

