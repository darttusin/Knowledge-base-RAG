RNN 
==========================================

*class* torch.nn. RNN ( *input_size*  , *hidden_size*  , *num_layers = 1*  , *nonlinearity = 'tanh'*  , *bias = True*  , *batch_first = False*  , *dropout = 0.0*  , *bidirectional = False*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/rnn.py#L470) 
:   Apply a multi-layer Elman RNN with <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            tanh
           </mi>
<mo>
            ⁡
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           tanh
          </annotation>
</semantics>
</math> -->tanh ⁡ tanhtanh  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            ReLU
           </mtext>
</mrow>
<annotation encoding="application/x-tex">
           text{ReLU}
          </annotation>
</semantics>
</math> -->ReLU text{ReLU}ReLU  non-linearity to an input sequence. For each element in the input sequence,
each layer computes the following function: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             h
            </mi>
<mi>
             t
            </mi>
</msub>
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
             x
            </mi>
<mi>
             t
            </mi>
</msub>
<msubsup>
<mi>
             W
            </mi>
<mrow>
<mi>
              i
             </mi>
<mi>
              h
             </mi>
</mrow>
<mi>
             T
            </mi>
</msubsup>
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
              h
             </mi>
</mrow>
</msub>
<mo>
            +
           </mo>
<msub>
<mi>
             h
            </mi>
<mrow>
<mi>
              t
             </mi>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
</mrow>
</msub>
<msubsup>
<mi>
             W
            </mi>
<mrow>
<mi>
              h
             </mi>
<mi>
              h
             </mi>
</mrow>
<mi>
             T
            </mi>
</msubsup>
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
              h
             </mi>
</mrow>
</msub>
<mo stretchy="false">
            )
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           h_t = tanh(x_t W_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T + b_{hh})
          </annotation>
</semantics>
</math> -->
h t = tanh ⁡ ( x t W i h T + b i h + h t − 1 W h h T + b h h ) h_t = tanh(x_t W_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T + b_{hh})

h t ​ = tanh ( x t ​ W ih T ​ + b ih ​ + h t − 1 ​ W hh T ​ + b hh ​ )

where <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             h
            </mi>
<mi>
             t
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           h_t
          </annotation>
</semantics>
</math> -->h t h_th t ​  is the hidden state at time *t* , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             x
            </mi>
<mi>
             t
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           x_t
          </annotation>
</semantics>
</math> -->x t x_tx t ​  is
the input at time *t* , and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             h
            </mi>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mi>
              t
             </mi>
<mo>
              −
             </mo>
<mn>
              1
             </mn>
<mo stretchy="false">
              )
             </mo>
</mrow>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           h_{(t-1)}
          </annotation>
</semantics>
</math> -->h ( t − 1 ) h_{(t-1)}h ( t − 1 ) ​  is the hidden state of the
previous layer at time *t-1* or the initial hidden state at time *0* .
If `nonlinearity`  is `'relu'`  , then <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mtext>
            ReLU
           </mtext>
</mrow>
<annotation encoding="application/x-tex">
           text{ReLU}
          </annotation>
</semantics>
</math> -->ReLU text{ReLU}ReLU  is used instead of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            tanh
           </mi>
<mo>
            ⁡
           </mo>
</mrow>
<annotation encoding="application/x-tex">
           tanh
          </annotation>
</semantics>
</math> -->tanh ⁡ tanhtanh  . 

```
# Efficient implementation equivalent to the following with bidirectional=False
rnn = nn.RNN(input_size, hidden_size, num_layers)
params = dict(rnn.named_parameters())
def forward(x, hx=None, batch_first=False):
    if batch_first:
        x = x.transpose(0, 1)
    seq_len, batch_size, _ = x.size()
    if hx is None:
        hx = torch.zeros(rnn.num_layers, batch_size, rnn.hidden_size)
    h_t_minus_1 = hx.clone()
    h_t = hx.clone()
    output = []
    for t in range(seq_len):
        for layer in range(rnn.num_layers):
            input_t = x[t] if layer == 0 else h_t[layer - 1]
            h_t[layer] = torch.tanh(
                input_t @ params[f"weight_ih_l{layer}"].T
                + h_t_minus_1[layer] @ params[f"weight_hh_l{layer}"].T
                + params[f"bias_hh_l{layer}"]
                + params[f"bias_ih_l{layer}"]
            )
        output.append(h_t[-1].clone())
        h_t_minus_1 = h_t.clone()
    output = torch.stack(output)
    if batch_first:
        output = output.transpose(0, 1)
    return output, h_t

```

Parameters
:   * **input_size** – The number of expected features in the input *x*
* **hidden_size** – The number of features in the hidden state *h*
* **num_layers** – Number of recurrent layers. E.g., setting `num_layers=2`  would mean stacking two RNNs together to form a *stacked RNN* ,
with the second RNN taking in outputs of the first RNN and
computing the final results. Default: 1
* **nonlinearity** – The non-linearity to use. Can be either `'tanh'`  or `'relu'`  . Default: `'tanh'`
* **bias** – If `False`  , then the layer does not use bias weights *b_ih* and *b_hh* .
Default: `True`
* **batch_first** – If `True`  , then the input and output tensors are provided
as *(batch, seq, feature)* instead of *(seq, batch, feature)* .
Note that this does not apply to hidden or cell states. See the
Inputs/Outputs sections below for details. Default: `False`
* **dropout** – If non-zero, introduces a *Dropout* layer on the outputs of each
RNN layer except the last layer, with dropout probability equal to `dropout`  . Default: 0
* **bidirectional** – If `True`  , becomes a bidirectional RNN. Default: `False`

Inputs: input, hx
:   * **input** : tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                L
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
               (L, H_{in})
              </annotation>
</semantics>
</math> -->( L , H i n ) (L, H_{in})( L , H in ​ )  for unbatched input, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                L
               </mi>
<mo separator="true">
                ,
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
               (L, N, H_{in})
              </annotation>
</semantics>
</math> -->( L , N , H i n ) (L, N, H_{in})( L , N , H in ​ )  when `batch_first=False`  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                L
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
               (N, L, H_{in})
              </annotation>
</semantics>
</math> -->( N , L , H i n ) (N, L, H_{in})( N , L , H in ​ )  when `batch_first=True`  containing the features of
the input sequence. The input can also be a packed variable length sequence.
See [`torch.nn.utils.rnn.pack_padded_sequence()`](torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence "torch.nn.utils.rnn.pack_padded_sequence")  or [`torch.nn.utils.rnn.pack_sequence()`](torch.nn.utils.rnn.pack_sequence.html#torch.nn.utils.rnn.pack_sequence "torch.nn.utils.rnn.pack_sequence")  for details.

* **hx** : tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                D
               </mi>
<mo>
                ∗
               </mo>
<mtext>
                num_layers
               </mtext>
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
               (D * text{num_layers}, H_{out})
              </annotation>
</semantics>
</math> -->( D ∗ num_layers , H o u t ) (D * text{num_layers}, H_{out})( D ∗ num_layers , H o u t ​ )  for unbatched input or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                D
               </mi>
<mo>
                ∗
               </mo>
<mtext>
                num_layers
               </mtext>
<mo separator="true">
                ,
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
               (D * text{num_layers}, N, H_{out})
              </annotation>
</semantics>
</math> -->( D ∗ num_layers , N , H o u t ) (D * text{num_layers}, N, H_{out})( D ∗ num_layers , N , H o u t ​ )  containing the initial hidden
state for the input sequence batch. Defaults to zeros if not provided.

where: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right left" columnspacing="0em" rowspacing="0.25em">
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
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<mi>
                  L
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
                  sequence length
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
                  D
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
<mn>
                  2
                 </mn>
<mtext>
                  if bidirectional=True otherwise
                 </mtext>
<mn>
                  1
                 </mn>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
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
                  input_size
                 </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
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
                  hidden_size
                 </mtext>
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
             begin{aligned}
    N ={} &amp; text{batch size} 
    L ={} &amp; text{sequence length} 
    D ={} &amp; 2 text{ if bidirectional=True otherwise } 1 
    H_{in} ={} &amp; text{input_size} 
    H_{out} ={} &amp; text{hidden_size}
end{aligned}
            </annotation>
</semantics>
</math> -->
N = batch size L = sequence length D = 2 if bidirectional=True otherwise 1 H i n = input_size H o u t = hidden_size begin{aligned}
 N ={} & text{batch size} 
 L ={} & text{sequence length} 
 D ={} & 2 text{ if bidirectional=True otherwise } 1 
 H_{in} ={} & text{input_size} 
 H_{out} ={} & text{hidden_size}
end{aligned}

N = L = D = H in ​ = H o u t ​ = ​ batch size sequence length 2 if bidirectional=True otherwise 1 input_size hidden_size ​

Outputs: output, h_n
:   * **output** : tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                L
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                D
               </mi>
<mo>
                ∗
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
               (L, D * H_{out})
              </annotation>
</semantics>
</math> -->( L , D ∗ H o u t ) (L, D * H_{out})( L , D ∗ H o u t ​ )  for unbatched input, <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                L
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                N
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                D
               </mi>
<mo>
                ∗
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
               (L, N, D * H_{out})
              </annotation>
</semantics>
</math> -->( L , N , D ∗ H o u t ) (L, N, D * H_{out})( L , N , D ∗ H o u t ​ )  when `batch_first=False`  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                L
               </mi>
<mo separator="true">
                ,
               </mo>
<mi>
                D
               </mi>
<mo>
                ∗
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
               (N, L, D * H_{out})
              </annotation>
</semantics>
</math> -->( N , L , D ∗ H o u t ) (N, L, D * H_{out})( N , L , D ∗ H o u t ​ )  when `batch_first=True`  containing the output features *(h_t)* from the last layer of the RNN, for each *t* . If a [`torch.nn.utils.rnn.PackedSequence`](torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence "torch.nn.utils.rnn.PackedSequence")  has been given as the input, the output
will also be a packed sequence.

* **h_n** : tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                D
               </mi>
<mo>
                ∗
               </mo>
<mtext>
                num_layers
               </mtext>
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
               (D * text{num_layers}, H_{out})
              </annotation>
</semantics>
</math> -->( D ∗ num_layers , H o u t ) (D * text{num_layers}, H_{out})( D ∗ num_layers , H o u t ​ )  for unbatched input or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
                (
               </mo>
<mi>
                D
               </mi>
<mo>
                ∗
               </mo>
<mtext>
                num_layers
               </mtext>
<mo separator="true">
                ,
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
               (D * text{num_layers}, N, H_{out})
              </annotation>
</semantics>
</math> -->( D ∗ num_layers , N , H o u t ) (D * text{num_layers}, N, H_{out})( D ∗ num_layers , N , H o u t ​ )  containing the final hidden state
for each element in the batch.

Variables
:   * **weight_ih_l[k]** – the learnable input-hidden weights of the k-th layer,
of shape *(hidden_size, input_size)* for *k = 0* . Otherwise, the shape is *(hidden_size, num_directions * hidden_size)*
* **weight_hh_l[k]** – the learnable hidden-hidden weights of the k-th layer,
of shape *(hidden_size, hidden_size)*
* **bias_ih_l[k]** – the learnable input-hidden bias of the k-th layer,
of shape *(hidden_size)*
* **bias_hh_l[k]** – the learnable hidden-hidden bias of the k-th layer,
of shape *(hidden_size)*

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

Note 

For bidirectional RNNs, forward and backward are directions 0 and 1 respectively.
Example of splitting the output layers when `batch_first=False`  : `output.view(seq_len, batch, num_directions, hidden_size)`  .

Note 

`batch_first`  argument is ignored for unbatched inputs.

Warning 

There are known non-determinism issues for RNN functions on some versions of cuDNN and CUDA.
You can enforce deterministic behavior by setting the following environment variables: 

On CUDA 10.1, set environment variable `CUDA_LAUNCH_BLOCKING=1`  .
This may affect performance. 

On CUDA 10.2 or later, set environment variable
(note the leading colon symbol) `CUBLAS_WORKSPACE_CONFIG=:16:8`  or `CUBLAS_WORKSPACE_CONFIG=:4096:2`  . 

See the [cuDNN 8 Release Notes](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-880/release-notes/rel_8.html)  for more information.

Note 

If the following conditions are satisfied:
1) cudnn is enabled,
2) input data is on the GPU
3) input data has dtype `torch.float16`  4) V100 GPU is used,
5) input data is not in `PackedSequence`  format
persistent algorithm can be selected to improve performance.

Examples: 

```
>>> rnn = nn.RNN(10, 20, 2)
>>> input = torch.randn(5, 3, 10)
>>> h0 = torch.randn(2, 3, 20)
>>> output, hn = rnn(input, h0)

```

