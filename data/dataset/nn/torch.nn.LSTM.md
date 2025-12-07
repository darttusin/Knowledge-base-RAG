LSTM 
============================================

*class* torch.nn. LSTM ( *input_size*  , *hidden_size*  , *num_layers = 1*  , *bias = True*  , *batch_first = False*  , *dropout = 0.0*  , *bidirectional = False*  , *proj_size = 0*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/rnn.py#L797) 
:   Apply a multi-layer long short-term memory (LSTM) RNN to an input sequence.
For each element in the input sequence, each layer computes the following
function: 

<!-- MathML: <math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mtable columnalign="left left" columnspacing="1em" rowspacing="0.16em">
<mtr>
<mtd>
<mstyle displaystyle="false" scriptlevel="0">
<mrow>
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
                 i
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
                  i
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
<msub>
<mi>
                 f
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
                  f
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
<msub>
<mi>
                 g
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
                  g
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
<msub>
<mi>
                 o
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
                  o
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
<msub>
<mi>
                 c
                </mi>
<mi>
                 t
                </mi>
</msub>
<mo>
                =
               </mo>
<msub>
<mi>
                 f
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
                 c
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
<mo>
                +
               </mo>
<msub>
<mi>
                 i
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
                 g
                </mi>
<mi>
                 t
                </mi>
</msub>
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
<msub>
<mi>
                 o
                </mi>
<mi>
                 t
                </mi>
</msub>
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
<msub>
<mi>
                 c
                </mi>
<mi>
                 t
                </mi>
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
           begin{array}{ll} 
    i_t = sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) 
    f_t = sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) 
    g_t = tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) 
    o_t = sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) 
    c_t = f_t odot c_{t-1} + i_t odot g_t 
    h_t = o_t odot tanh(c_t) 
end{array}
          </annotation>
</semantics>
</math> -->
i t = σ ( W i i x t + b i i + W h i h t − 1 + b h i ) f t = σ ( W i f x t + b i f + W h f h t − 1 + b h f ) g t = tanh ⁡ ( W i g x t + b i g + W h g h t − 1 + b h g ) o t = σ ( W i o x t + b i o + W h o h t − 1 + b h o ) c t = f t ⊙ c t − 1 + i t ⊙ g t h t = o t ⊙ tanh ⁡ ( c t ) begin{array}{ll} 
 i_t = sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) 
 f_t = sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) 
 g_t = tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) 
 o_t = sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) 
 c_t = f_t odot c_{t-1} + i_t odot g_t 
 h_t = o_t odot tanh(c_t) 
end{array}

i t ​ = σ ( W ii ​ x t ​ + b ii ​ + W hi ​ h t − 1 ​ + b hi ​ ) f t ​ = σ ( W i f ​ x t ​ + b i f ​ + W h f ​ h t − 1 ​ + b h f ​ ) g t ​ = tanh ( W i g ​ x t ​ + b i g ​ + W h g ​ h t − 1 ​ + b h g ​ ) o t ​ = σ ( W i o ​ x t ​ + b i o ​ + W h o ​ h t − 1 ​ + b h o ​ ) c t ​ = f t ​ ⊙ c t − 1 ​ + i t ​ ⊙ g t ​ h t ​ = o t ​ ⊙ tanh ( c t ​ ) ​

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
             c
            </mi>
<mi>
             t
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           c_t
          </annotation>
</semantics>
</math> -->c t c_tc t ​  is the cell
state at time *t* , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->x t x_tx t ​  is the input at time *t* , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
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
</mrow>
<annotation encoding="application/x-tex">
           h_{t-1}
          </annotation>
</semantics>
</math> -->h t − 1 h_{t-1}h t − 1 ​  is the hidden state of the layer at time *t-1* or the initial hidden
state at time *0* , and <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             i
            </mi>
<mi>
             t
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           i_t
          </annotation>
</semantics>
</math> -->i t i_ti t ​  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             f
            </mi>
<mi>
             t
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           f_t
          </annotation>
</semantics>
</math> -->f t f_tf t ​  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             g
            </mi>
<mi>
             t
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           g_t
          </annotation>
</semantics>
</math> -->g t g_tg t ​  , <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
<semantics>
<mrow>
<msub>
<mi>
             o
            </mi>
<mi>
             t
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           o_t
          </annotation>
</semantics>
</math> -->o t o_to t ​  are the input, forget, cell, and output gates, respectively. <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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

In a multilayer LSTM, the input <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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

If `proj_size > 0`  is specified, LSTM with projections will be used. This changes
the LSTM cell in the following way. First, the dimension of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> -->h t h_th t ​  will be changed from `hidden_size`  to `proj_size`  (dimensions of <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
              i
             </mi>
</mrow>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           W_{hi}
          </annotation>
</semantics>
</math> -->W h i W_{hi}W hi ​  will be changed accordingly).
Second, the output hidden state of each layer will be multiplied by a learnable projection
matrix: <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
<mi>
             t
            </mi>
</msub>
</mrow>
<annotation encoding="application/x-tex">
           h_t = W_{hr}h_t
          </annotation>
</semantics>
</math> -->h t = W h r h t h_t = W_{hr}h_th t ​ = W h r ​ h t ​  . Note that as a consequence of this, the output
of LSTM network will be of different shape as well. See Inputs/Outputs sections below for exact
dimensions of all variables. You can find more details in [https://arxiv.org/abs/1402.1128](https://arxiv.org/abs/1402.1128)  . 

Parameters
:   * **input_size** – The number of expected features in the input *x*
* **hidden_size** – The number of features in the hidden state *h*
* **num_layers** – Number of recurrent layers. E.g., setting `num_layers=2`  would mean stacking two LSTMs together to form a *stacked LSTM* ,
with the second LSTM taking in outputs of the first LSTM and
computing the final results. Default: 1
* **bias** – If `False`  , then the layer does not use bias weights *b_ih* and *b_hh* .
Default: `True`
* **batch_first** – If `True`  , then the input and output tensors are provided
as *(batch, seq, feature)* instead of *(seq, batch, feature)* .
Note that this does not apply to hidden or cell states. See the
Inputs/Outputs sections below for details. Default: `False`
* **dropout** – If non-zero, introduces a *Dropout* layer on the outputs of each
LSTM layer except the last layer, with dropout probability equal to `dropout`  . Default: 0
* **bidirectional** – If `True`  , becomes a bidirectional LSTM. Default: `False`
* **proj_size** – If `> 0`  , will use LSTM with projections of corresponding size. Default: 0

Inputs: input, (h_0, c_0)
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
</math> -->( D ∗ num_layers , N , H o u t ) (D * text{num_layers}, N, H_{out})( D ∗ num_layers , N , H o u t ​ )  containing the
initial hidden state for each element in the input sequence.
Defaults to zeros if (h_0, c_0) is not provided.

* **c_0** : tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                  c
                 </mi>
<mi>
                  e
                 </mi>
<mi>
                  l
                 </mi>
<mi>
                  l
                 </mi>
</mrow>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (D * text{num_layers}, H_{cell})
              </annotation>
</semantics>
</math> -->( D ∗ num_layers , H c e l l ) (D * text{num_layers}, H_{cell})( D ∗ num_layers , H ce ll ​ )  for unbatched input or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                  c
                 </mi>
<mi>
                  e
                 </mi>
<mi>
                  l
                 </mi>
<mi>
                  l
                 </mi>
</mrow>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (D * text{num_layers}, N, H_{cell})
              </annotation>
</semantics>
</math> -->( D ∗ num_layers , N , H c e l l ) (D * text{num_layers}, N, H_{cell})( D ∗ num_layers , N , H ce ll ​ )  containing the
initial cell state for each element in the input sequence.
Defaults to zeros if (h_0, c_0) is not provided.

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
                    c
                   </mi>
<mi>
                    e
                   </mi>
<mi>
                    l
                   </mi>
<mi>
                    l
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
                  proj_size if proj_size
                 </mtext>
<mo>
                  &gt;
                 </mo>
<mn>
                  0
                 </mn>
<mtext>
                  otherwise hidden_size
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
    H_{cell} ={} &amp; text{hidden_size} 
    H_{out} ={} &amp; text{proj_size if } text{proj_size}&gt;0 text{ otherwise hidden_size} 
end{aligned}
            </annotation>
</semantics>
</math> -->
N = batch size L = sequence length D = 2 if bidirectional=True otherwise 1 H i n = input_size H c e l l = hidden_size H o u t = proj_size if proj_size > 0 otherwise hidden_size begin{aligned}
 N ={} & text{batch size} 
 L ={} & text{sequence length} 
 D ={} & 2 text{ if bidirectional=True otherwise } 1 
 H_{in} ={} & text{input_size} 
 H_{cell} ={} & text{hidden_size} 
 H_{out} ={} & text{proj_size if } text{proj_size}>0 text{ otherwise hidden_size} 
end{aligned}

N = L = D = H in ​ = H ce ll ​ = H o u t ​ = ​ batch size sequence length 2 if bidirectional=True otherwise 1 input_size hidden_size proj_size if proj_size > 0 otherwise hidden_size ​

Outputs: output, (h_n, c_n)
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
</math> -->( N , L , D ∗ H o u t ) (N, L, D * H_{out})( N , L , D ∗ H o u t ​ )  when `batch_first=True`  containing the output features *(h_t)* from the last layer of the LSTM, for each *t* . If a [`torch.nn.utils.rnn.PackedSequence`](torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence "torch.nn.utils.rnn.PackedSequence")  has been given as the input, the output
will also be a packed sequence. When `bidirectional=True`  , *output* will contain
a concatenation of the forward and reverse hidden states at each time step in the sequence.

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
</math> -->( D ∗ num_layers , N , H o u t ) (D * text{num_layers}, N, H_{out})( D ∗ num_layers , N , H o u t ​ )  containing the
final hidden state for each element in the sequence. When `bidirectional=True`  , *h_n* will contain a concatenation of the final forward and reverse hidden states, respectively.

* **c_n** : tensor of shape <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                  c
                 </mi>
<mi>
                  e
                 </mi>
<mi>
                  l
                 </mi>
<mi>
                  l
                 </mi>
</mrow>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (D * text{num_layers}, H_{cell})
              </annotation>
</semantics>
</math> -->( D ∗ num_layers , H c e l l ) (D * text{num_layers}, H_{cell})( D ∗ num_layers , H ce ll ​ )  for unbatched input or <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
                  c
                 </mi>
<mi>
                  e
                 </mi>
<mi>
                  l
                 </mi>
<mi>
                  l
                 </mi>
</mrow>
</msub>
<mo stretchy="false">
                )
               </mo>
</mrow>
<annotation encoding="application/x-tex">
               (D * text{num_layers}, N, H_{cell})
              </annotation>
</semantics>
</math> -->( D ∗ num_layers , N , H c e l l ) (D * text{num_layers}, N, H_{cell})( D ∗ num_layers , N , H ce ll ​ )  containing the
final cell state for each element in the sequence. When `bidirectional=True`  , *c_n* will contain a concatenation of the final forward and reverse cell states, respectively.

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
</math> -->k t h text{k}^{th}k t h  layer *(W_ii|W_if|W_ig|W_io)* , of shape *(4*hidden_size, input_size)* for *k = 0* .
Otherwise, the shape is *(4*hidden_size, num_directions * hidden_size)* . If `proj_size > 0`  was specified, the shape will be *(4*hidden_size, num_directions * proj_size)* for *k > 0*

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
</math> -->k t h text{k}^{th}k t h  layer *(W_hi|W_hf|W_hg|W_ho)* , of shape *(4*hidden_size, hidden_size)* . If `proj_size > 0`  was specified, the shape will be *(4*hidden_size, proj_size)* .

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
</math> -->k t h text{k}^{th}k t h  layer *(b_ii|b_if|b_ig|b_io)* , of shape *(4*hidden_size)*

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
</math> -->k t h text{k}^{th}k t h  layer *(b_hi|b_hf|b_hg|b_ho)* , of shape *(4*hidden_size)*

* **weight_hr_l[k]** – the learnable projection weights of the <!-- MathML: <math xmlns="http://www.w3.org/1998/Math/MathML">
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
of shape *(proj_size, hidden_size)* . Only present when `proj_size > 0`  was
specified.

* **weight_ih_l[k]_reverse** – Analogous to *weight_ih_l[k]* for the reverse direction.
Only present when `bidirectional=True`  .
* **weight_hh_l[k]_reverse** – Analogous to *weight_hh_l[k]* for the reverse direction.
Only present when `bidirectional=True`  .
* **bias_ih_l[k]_reverse** – Analogous to *bias_ih_l[k]* for the reverse direction.
Only present when `bidirectional=True`  .
* **bias_hh_l[k]_reverse** – Analogous to *bias_hh_l[k]* for the reverse direction.
Only present when `bidirectional=True`  .
* **weight_hr_l[k]_reverse** – Analogous to *weight_hr_l[k]* for the reverse direction.
Only present when `bidirectional=True`  and `proj_size > 0`  was specified.

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

For bidirectional LSTMs, forward and backward are directions 0 and 1 respectively.
Example of splitting the output layers when `batch_first=False`  : `output.view(seq_len, batch, num_directions, hidden_size)`  .

Note 

For bidirectional LSTMs, *h_n* is not equivalent to the last element of *output* ; the
former contains the final forward and reverse hidden states, while the latter contains the
final forward hidden state and the initial reverse hidden state.

Note 

`batch_first`  argument is ignored for unbatched inputs.

Note 

`proj_size`  should be smaller than `hidden_size`  .

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
>>> rnn = nn.LSTM(10, 20, 2)
>>> input = torch.randn(5, 3, 10)
>>> h0 = torch.randn(2, 3, 20)
>>> c0 = torch.randn(2, 3, 20)
>>> output, (hn, cn) = rnn(input, (h0, c0))

```

