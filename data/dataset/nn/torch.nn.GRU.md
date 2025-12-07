GRU 
==========================================

*class* torch.nn. GRU ( *input_size*  , *hidden_size*  , *num_layers = 1*  , *bias = True*  , *batch_first = False*  , *dropout = 0.0*  , *bidirectional = False*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/rnn.py#L1162) 
:   Apply a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
For each element in the input sequence, each layer computes the following
function: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="left left" columnspacing="1em" rowspacing="0.16em">
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
<msub>
<mi>
                 r
                </mi>
<mi>
                 t
                </mi>
</msub>
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
<msub>
<mi>
                 x
                </mi>
<mi>
                 t
                </mi>
</msub>
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
<msub>
<mi>
                 z
                </mi>
<mi>
                 t
                </mi>
</msub>
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
<msub>
<mi>
                 x
                </mi>
<mi>
                 t
                </mi>
</msub>
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
<msub>
<mi>
                 n
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
<msub>
<mi>
                 x
                </mi>
<mi>
                 t
                </mi>
</msub>
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
<msub>
<mi>
                 r
                </mi>
<mi>
                 t
                </mi>
</msub>
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
                 z
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo stretchy="false">
                )
               </mo>
<mo>
                ⊙
               </mo>
<msub>
<mi>
                 n
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                +
               </mo>
<msub>
<mi>
                 z
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                ⊙
               </mo>
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
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
           begin{array}{ll}
    r_t = sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) 
    z_t = sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) 
    n_t = tanh(W_{in} x_t + b_{in} + r_t odot (W_{hn} h_{(t-1)}+ b_{hn})) 
    h_t = (1 - z_t) odot n_t + z_t odot h_{(t-1)}
end{array}
          </annotation>
</semantics>
</math> -->
r t = σ ( W i r x t + b i r + W h r h ( t − 1 ) + b h r ) z t = σ ( W i z x t + b i z + W h z h ( t − 1 ) + b h z ) n t = tanh ⁡ ( W i n x t + b i n + r t ⊙ ( W h n h ( t − 1 ) + b h n ) ) h t = ( 1 − z t ) ⊙ n t + z t ⊙ h ( t − 1 ) begin{array}{ll}
 r_t = sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) 
 z_t = sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) 
 n_t = tanh(W_{in} x_t + b_{in} + r_t odot (W_{hn} h_{(t-1)}+ b_{hn})) 
 h_t = (1 - z_t) odot n_t + z_t odot h_{(t-1)}
end{array}

r t ​ = σ ( W i r ​ x t ​ + b i r ​ + W h r ​ h ( t − 1 ) ​ + b h r ​ ) z t ​ = σ ( W i z ​ x t ​ + b i z ​ + W h z ​ h ( t − 1 ) ​ + b h z ​ ) n t ​ = tanh ( W in ​ x t ​ + b in ​ + r t ​ ⊙ ( W hn ​ h ( t − 1 ) ​ + b hn ​ )) h t ​ = ( 1 − z t ​ ) ⊙ n t ​ + z t ​ ⊙ h ( t − 1 ) ​ ​

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
</math> -->x t x_tx t ​  is the input
at time *t* , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->h ( t − 1 ) h_{(t-1)}h ( t − 1 ) ​  is the hidden state of the layer
at time *t-1* or the initial hidden state at time *0* , and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             r
            </mi>
<mi>
             t
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           r_t
          </annotation>
</semantics>
</math> -->r t r_tr t ​  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             z
            </mi>
<mi>
             t
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           z_t
          </annotation>
</semantics>
</math> -->z t z_tz t ​  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             n
            </mi>
<mi>
             t
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           n_t
          </annotation>
</semantics>
</math> -->n t n_tn t ​  are the reset, update, and new gates, respectively. <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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

In a multilayer GRU, the input <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msubsup>
<mi>
             x
            </mi>
<mi>
             t
            </mi>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mi>
              l
             </mi>
<mo stretchy="false">
              )
             </mo>
</mrow>
</msubsup>
</mrow>
<annotation encoding="application/x-tex">
           x^{(l)}_t
          </annotation>
</semantics>
</math> -->x t ( l ) x^{(l)}_tx t ( l ) ​  of the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            l
           </mi>
</mrow>
<annotation encoding="application/x-tex">
           l
          </annotation>
</semantics>
</math> -->l ll  -th layer
( <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mi>
            l
           </mi>
<mo>
            ≥
           </mo>
<mn>
            2
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           l ge 2
          </annotation>
</semantics>
</math> -->l ≥ 2 l ge 2l ≥ 2  ) is the hidden state <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msubsup>
<mi>
             h
            </mi>
<mi>
             t
            </mi>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mi>
              l
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
</msubsup>
</mrow>
<annotation encoding="application/x-tex">
           h^{(l-1)}_t
          </annotation>
</semantics>
</math> -->h t ( l − 1 ) h^{(l-1)}_th t ( l − 1 ) ​  of the previous layer multiplied by
dropout <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msubsup>
<mi>
             δ
            </mi>
<mi>
             t
            </mi>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mi>
              l
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
</msubsup>
</mrow>
<annotation encoding="application/x-tex">
           delta^{(l-1)}_t
          </annotation>
</semantics>
</math> -->δ t ( l − 1 ) delta^{(l-1)}_tδ t ( l − 1 ) ​  where each <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msubsup>
<mi>
             δ
            </mi>
<mi>
             t
            </mi>
<mrow>
<mo stretchy="false">
              (
             </mo>
<mi>
              l
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
</msubsup>
</mrow>
<annotation encoding="application/x-tex">
           delta^{(l-1)}_t
          </annotation>
</semantics>
</math> -->δ t ( l − 1 ) delta^{(l-1)}_tδ t ( l − 1 ) ​  is a Bernoulli random
variable which is <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mn>
            0
           </mn>
</mrow>
<annotation encoding="application/x-tex">
           0
          </annotation>
</semantics>
</math> -->0 00  with probability `dropout`  . 

Parameters
:   * **input_size** – The number of expected features in the input *x*
* **hidden_size** – The number of features in the hidden state *h*
* **num_layers** – Number of recurrent layers. E.g., setting `num_layers=2`  would mean stacking two GRUs together to form a *stacked GRU* ,
with the second GRU taking in outputs of the first GRU and
computing the final results. Default: 1
* **bias** – If `False`  , then the layer does not use bias weights *b_ih* and *b_hh* .
Default: `True`
* **batch_first** – If `True`  , then the input and output tensors are provided
as *(batch, seq, feature)* instead of *(seq, batch, feature)* .
Note that this does not apply to hidden or cell states. See the
Inputs/Outputs sections below for details. Default: `False`
* **dropout** – If non-zero, introduces a *Dropout* layer on the outputs of each
GRU layer except the last layer, with dropout probability equal to `dropout`  . Default: 0
* **bidirectional** – If `True`  , becomes a bidirectional GRU. Default: `False`

Inputs: input, h_0
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

* **h_0** : tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( D ∗ num_layers , H o u t ) (D * text{num_layers}, H_{out})( D ∗ num_layers , H o u t ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->( D ∗ num_layers , N , H o u t ) (D * text{num_layers}, N, H_{out})( D ∗ num_layers , N , H o u t ​ )  containing the initial hidden state for the input sequence. Defaults to zeros if not provided.

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
</math> -->( N , L , D ∗ H o u t ) (N, L, D * H_{out})( N , L , D ∗ H o u t ​ )  when `batch_first=True`  containing the output features *(h_t)* from the last layer of the GRU, for each *t* . If a [`torch.nn.utils.rnn.PackedSequence`](torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence "torch.nn.utils.rnn.PackedSequence")  has been given as the input, the output
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
</math> -->( D ∗ num_layers , H o u t ) (D * text{num_layers}, H_{out})( D ∗ num_layers , H o u t ​ )  or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
for the input sequence.

Variables
:   * **weight_ih_l[k]** – the learnable input-hidden weights of the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mtext>
                 k
                </mtext>
<mrow>
<mi>
                  t
                 </mi>
<mi>
                  h
                 </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
               text{k}^{th}
              </annotation>
</semantics>
</math> -->k t h text{k}^{th}k t h  layer
(W_ir|W_iz|W_in), of shape *(3*hidden_size, input_size)* for *k = 0* .
Otherwise, the shape is *(3*hidden_size, num_directions * hidden_size)*

* **weight_hh_l[k]** – the learnable hidden-hidden weights of the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mtext>
                 k
                </mtext>
<mrow>
<mi>
                  t
                 </mi>
<mi>
                  h
                 </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
               text{k}^{th}
              </annotation>
</semantics>
</math> -->k t h text{k}^{th}k t h  layer
(W_hr|W_hz|W_hn), of shape *(3*hidden_size, hidden_size)*

* **bias_ih_l[k]** – the learnable input-hidden bias of the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mtext>
                 k
                </mtext>
<mrow>
<mi>
                  t
                 </mi>
<mi>
                  h
                 </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
               text{k}^{th}
              </annotation>
</semantics>
</math> -->k t h text{k}^{th}k t h  layer
(b_ir|b_iz|b_in), of shape *(3*hidden_size)*

* **bias_hh_l[k]** – the learnable hidden-hidden bias of the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msup>
<mtext>
                 k
                </mtext>
<mrow>
<mi>
                  t
                 </mi>
<mi>
                  h
                 </mi>
</mrow>
</msup>
</mrow>
<annotation encoding="application/x-tex">
               text{k}^{th}
              </annotation>
</semantics>
</math> -->k t h text{k}^{th}k t h  layer
(b_hr|b_hz|b_hn), of shape *(3*hidden_size)*

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

For bidirectional GRUs, forward and backward are directions 0 and 1 respectively.
Example of splitting the output layers when `batch_first=False`  : `output.view(seq_len, batch, num_directions, hidden_size)`  .

Note 

`batch_first`  argument is ignored for unbatched inputs.

Note 

The calculation of new gate <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              n
             </mi>
<mi>
              t
             </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            n_t
           </annotation>
</semantics>
</math> -->n t n_tn t ​  subtly differs from the original paper and other frameworks.
In the original implementation, the Hadamard product <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<mo stretchy="false">
             (
            </mo>
<mo>
             ⊙
            </mo>
<mo stretchy="false">
             )
            </mo>
</mrow>
<annotation encoding="application/x-tex">
            (odot)
           </annotation>
</semantics>
</math> -->( ⊙ ) (odot)( ⊙ )  between <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
              r
             </mi>
<mi>
              t
             </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
            r_t
           </annotation>
</semantics>
</math> -->r t r_tr t ​  and the
previous hidden state <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->h ( t − 1 ) h_{(t-1)}h ( t − 1 ) ​  is done before the multiplication with the weight matrix *W* and addition of bias: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right" columnspacing="" rowspacing="0.25em">
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<msub>
<mi>
                  n
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
<msub>
<mi>
                  x
                 </mi>
<mi>
                  t
                 </mi>
</msub>
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
<mo stretchy="false">
                 (
                </mo>
<msub>
<mi>
                  r
                 </mi>
<mi>
                  t
                 </mi>
</msub>
<mo>
                 ⊙
                </mo>
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
<mo stretchy="false">
                 )
                </mo>
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
</mrow>
</mstyle>
</mtd>
</mtr>
</mtable>
<annotation encoding="application/x-tex">
            begin{aligned}
    n_t = tanh(W_{in} x_t + b_{in} + W_{hn} ( r_t odot h_{(t-1)} ) + b_{hn})
end{aligned}
           </annotation>
</semantics>
</math> -->
n t = tanh ⁡ ( W i n x t + b i n + W h n ( r t ⊙ h ( t − 1 ) ) + b h n ) begin{aligned}
 n_t = tanh(W_{in} x_t + b_{in} + W_{hn} ( r_t odot h_{(t-1)} ) + b_{hn})
end{aligned}

n t ​ = tanh ( W in ​ x t ​ + b in ​ + W hn ​ ( r t ​ ⊙ h ( t − 1 ) ​ ) + b hn ​ ) ​

This is in contrast to PyTorch implementation, which is done after <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
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
            W_{hn} h_{(t-1)}
           </annotation>
</semantics>
</math> -->W h n h ( t − 1 ) W_{hn} h_{(t-1)}W hn ​ h ( t − 1 ) ​ 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="right" columnspacing="" rowspacing="0.25em">
<mtr>
<mtd>
<mstyle displaystyle="true" scriptlevel="0">
<mrow>
<msub>
<mi>
                  n
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
<msub>
<mi>
                  x
                 </mi>
<mi>
                  t
                 </mi>
</msub>
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
<msub>
<mi>
                  r
                 </mi>
<mi>
                  t
                 </mi>
</msub>
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
</mtable>
<annotation encoding="application/x-tex">
            begin{aligned}
    n_t = tanh(W_{in} x_t + b_{in} + r_t odot (W_{hn} h_{(t-1)}+ b_{hn}))
end{aligned}
           </annotation>
</semantics>
</math> -->
n t = tanh ⁡ ( W i n x t + b i n + r t ⊙ ( W h n h ( t − 1 ) + b h n ) ) begin{aligned}
 n_t = tanh(W_{in} x_t + b_{in} + r_t odot (W_{hn} h_{(t-1)}+ b_{hn}))
end{aligned}

n t ​ = tanh ( W in ​ x t ​ + b in ​ + r t ​ ⊙ ( W hn ​ h ( t − 1 ) ​ + b hn ​ )) ​

This implementation differs on purpose for efficiency.

Note 

If the following conditions are satisfied:
1) cudnn is enabled,
2) input data is on the GPU
3) input data has dtype `torch.float16`  4) V100 GPU is used,
5) input data is not in `PackedSequence`  format
persistent algorithm can be selected to improve performance.

Examples: 

```
>>> rnn = nn.GRU(10, 20, 2)
>>> input = torch.randn(5, 3, 10)
>>> h0 = torch.randn(2, 3, 20)
>>> output, hn = rnn(input, h0)

```

